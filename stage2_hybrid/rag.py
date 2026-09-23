"""
المرحلة 2: استرجاع متعدد المصادر + Hybrid + Reranking + معالجة الفجوة اللغوية
------------------------------------------------------------------------------
نضيف هنا فوق المرحلة 1 ثلاث تحسينات تعالج بالضبط الفجوات التي وثّقتها اختبارات
المرحلة 1:

1. مصدران مدمجان في فهرس واحد: مقالات KB الرسمية + التذاكر السابقة المحلولة.
   الآن سؤال مثل "الكود بتاع السحب خلص عليا" يستطيع أن يجد تذكرة 1129 حتى لو لم
   توجد صياغة مطابقة في أي مقالة KB.
2. Hybrid retrieval (TF-IDF + BM25 عبر RRF) لنفس الأسباب المعتادة.
3. ترجمة/توسيع السؤال عبر LLM قبل البحث (Query Translation)، فقط إذا توفر
   HF_API_TOKEN أو OPENROUTER_API_KEY: نطلب من النموذج ترجمة سؤال العميل للغة الأخرى، ثم نبحث
   بالنسختين معًا وندمج النتائج. هذا يعالج الفجوة اللغوية الحقيقية التي رصدناها
   (سؤال إنجليزي عن الرسوم لا يصل لمقالة عربية بنفس المعنى). بدون مفتاح API، هذه
   الخطوة تُتجاوز والفجوة تبقى موثّقة (راجع اختبارات المرحلة 3).

شغّله من داخل مجلد agent-assist-copilot:
    python -m stage2_hybrid.rag "الكود بتاع السحب من الوكيل خلص عليا"
"""

import logging
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
from rank_bm25 import BM25Okapi

from common import (
    load_kb, load_past_tickets, load_kb_large, load_past_tickets_large,
    build_kb_chunks, build_ticket_chunks, rrf_fuse, generate_draft,
    get_hf_client, get_openrouter_client, get_openrouter_model,
)
from offline_embeddings import (
    OfflineEmbedder, CharEmbedder, get_dense_embedder, normalize_for_index, normalize_arabic, concepts_of,
)

# AGENT_ASSIST_LARGE_DATA=1 يضيف مجموعة البيانات الصناعية الكبيرة (data/*_large.json)
# فوق البيانات الأصلية عند بناء الفهرس — لاختبار stage4 API على حجم بيانات أكبر
# بكثير دون المساس بالبيانات الأصلية أو كسر اختبارات pytest الحالية (المُعطَّل
# افتراضيًا، فالسلوك الافتراضي يبقى كما هو تمامًا).
USE_LARGE_DATA = os.environ.get("AGENT_ASSIST_LARGE_DATA", "").lower() in ("1", "true", "yes")

STORE_DIR = Path(__file__).parent / ("store_large" if USE_LARGE_DATA else "store")
INDEX_PATH = STORE_DIR / "index_v2.pkl"

CANDIDATE_POOL = 8
FINAL_K = 4

# الفهرس يُحمَّل مرة واحدة في الذاكرة (كان يُقرأ من القرص مع كل طلب).
_INDEX: dict | None = None


def _tokenize(text: str):
    """توكنز BM25: النص الموحّد + رموز المفاهيم ثنائية اللغة/اللهجية."""
    return normalize_for_index(text).split()


def build_index():
    global _INDEX
    STORE_DIR.mkdir(exist_ok=True)
    kb = load_kb()
    tickets = load_past_tickets()
    if USE_LARGE_DATA:
        kb = kb + load_kb_large()
        tickets = tickets + load_past_tickets_large()
    chunks = build_kb_chunks(kb) + build_ticket_chunks(tickets)
    texts = [c["embed_text"] for c in chunks]

    word = OfflineEmbedder()
    char = CharEmbedder()
    index = {
        "chunks": chunks,
        "word": word, "word_m": word.fit_transform(texts),
        "char": char, "char_m": char.fit_transform(texts),
        "bm25": BM25Okapi([_tokenize(t) for t in texts]),
        "dense": None, "dense_m": None,
    }
    index["concepts"] = [concepts_of(t) for t in texts]
    df = {}
    for cs in index["concepts"]:
        for c in cs:
            df[c] = df.get(c, 0) + 1
    index["concept_idf"] = {c: float(np.log((1 + len(texts)) / (1 + n)) + 1) for c, n in df.items()}
    dense = get_dense_embedder()
    if dense is not None:
        index["dense_m"] = dense.encode(texts)
        index["dense_name"] = dense.model_name

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({k: v for k, v in index.items() if k != "dense"}, f)
    index["dense"] = dense
    _INDEX = index

    n_kb = sum(1 for c in chunks if c["source_type"] == "kb")
    n_ticket = sum(1 for c in chunks if c["source_type"] == "past_ticket")
    mode = " [وضع البيانات الكبيرة مفعّل]" if USE_LARGE_DATA else ""
    dense_note = f" + dense ({index['dense_name']})" if dense is not None else ""
    print(f"تمت فهرسة {len(chunks)} قطعة ({n_kb} من KB + {n_ticket} من تذاكر سابقة){dense_note}.{mode}")
    return index


def load_index() -> dict:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if not INDEX_PATH.exists():
        return build_index()
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    index["dense"] = get_dense_embedder() if index.get("dense_m") is not None else None
    _INDEX = index
    return index


def translate_query_for_retrieval(question: str) -> str | None:
    """يطلب من نموذج LLM ترجمة السؤال للغة الأخرى (عربي↔إنجليزي) لتوسيع البحث
    عبر قاعدة معرفة ثنائية اللغة. مهمة "خفيفة" (رد قصير جدًا) — مُسنَدة أساسًا
    لـ Hugging Face، مع OpenRouter كمزوّد احتياطي يستلمها لو HF فشل. بدون أي
    مزوّد متاح يُتجاوز (يعيد None) وتبقى الفجوة اللغوية كما هي — موثّقة صراحة
    في تقييم المرحلة 3."""
    prompt = (
        f"ترجم النص التالي إلى العربية إن كان إنجليزيًا، أو إلى الإنجليزية إن كان عربيًا. "
        f"أعد الترجمة فقط بدون أي شرح إضافي:\n\n{question}"
    )

    hf_client = get_hf_client()
    if hf_client is not None:
        try:
            response = hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # فشل الترجمة يعني فقط بقاء الفجوة اللغوية كما هي (نفس سلوك عدم
            # وجود مفتاح أصلًا) — لا يجب أن يُسقط طلب /draft بالكامل.
            logging.getLogger("agent_copilot").warning("فشل استدعاء Hugging Face لترجمة السؤال، جارٍ تسليم المهمة لـ OpenRouter", exc_info=True)

    or_client = get_openrouter_client()
    if or_client is not None:
        try:
            response = or_client.chat.completions.create(
                model=get_openrouter_model(),
                messages=[{"role": "user", "content": prompt}], max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logging.getLogger("agent_copilot").warning("فشل استدعاء OpenRouter لترجمة السؤال", exc_info=True)

    return None


RANKERS = ("word", "char", "bm25", "concept", "dense")


def _concept_scores(index: dict, q: str) -> np.ndarray:
    """تطابق مفاهيم مرجَّح بالـ IDF: نسبة "وزن" مفاهيم السؤال الموجودة في المصدر.
    الإشارة الوحيدة المستقلة عن اللغة/اللهجة في المؤشرات المحلية (سؤال إنجليزي
    "account frozen" ومقالة "تجميد الحساب" يشتركان في __freeze__)."""
    idf = index["concept_idf"]
    q_concepts = concepts_of(q)
    total = sum(idf.get(c, 0.0) for c in q_concepts)
    if total == 0:
        return np.zeros(len(index["chunks"]))
    return np.array([sum(idf.get(c, 0.0) for c in q_concepts & cs) / total for cs in index["concepts"]])


def _score_all(index: dict, q: str) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity
    scores = {
        "word": cosine_similarity(index["word"].transform([q]), index["word_m"])[0],
        "char": cosine_similarity(index["char"].transform([q]), index["char_m"])[0],
        # BM25 قد يعطي درجات سالبة في مجموعة صغيرة (idf سالب للكلمات الشائعة).
        "bm25": np.maximum(index["bm25"].get_scores(_tokenize(q)), 0.0),
        "concept": _concept_scores(index, q),
    }
    if index.get("dense") is not None:
        qv = index["dense"].encode([q], is_query=True)[0]
        scores["dense"] = index["dense_m"] @ qv
    return scores


def retrieve_with_signals(question: str, pool: int = CANDIDATE_POOL, use_translation: bool = True,
                          teams: set[str] | None = None):
    """الاسترجاع الهجين + الإشارات الخام لكل مرشح (تُستخدم لحساب الثقة المعايَرة).

    teams: الأقسام المسموح بها. الفلترة تتم *قبل* الترتيب على مستوى الفهرس كله،
    فلا يمكن لمصدر من قسم آخر أن يظهر كمرشح أو في الـ prompt، ولا يوجد أي
    fallback لأقسام أخرى عند غياب النتائج (يُرجَع قائمة فارغة → تصعيد).
    teams=None يعني بدون تقييد (استخدام داخلي/تقييم فقط).

    يُرجع (candidates, signals) حيث signals[chunk_id] = أعلى درجة لكل مؤشر عبر
    صيغ السؤال + rrf + top3_frac (نسبة المؤشرات التي وضعته في أول 3)،
    و signals["__rrf_sorted__"] = درجات RRF مرتبة (لحساب الهامش)."""
    index = load_index()
    chunks = index["chunks"]
    allowed = np.array([teams is None or c["team"] in teams for c in chunks])
    if not allowed.any():
        return [], {}
    allowed_idx = np.flatnonzero(allowed)

    queries = [question]
    if use_translation:
        translated = translate_query_for_retrieval(question)
        if translated:
            queries.append(translated)

    # الإشارات (للثقة المعايَرة) من السؤال الأصلي فقط — نفس ظروف تدريب نموذج
    # الثقة، فلا تتغير الثقة حسب ناتج ترجمة LLM غير الحتمية. الترجمة تُضاف فقط
    # لترتيب المرشحين (توسيع الاسترجاع عبر اللغتين).
    best = {chunks[i]["chunk_id"]: {r: 0.0 for r in RANKERS} for i in allowed_idx}
    top3 = {cid: 0 for cid in best}
    original_rankings, rankings = [], []
    for qi, q in enumerate(queries):
        for name, s in _score_all(index, q).items():
            order = allowed_idx[np.argsort(-s[allowed_idx])]
            ranking = [chunks[i]["chunk_id"] for i in order]
            rankings.append(ranking)
            if qi > 0:
                continue
            original_rankings.append(ranking)
            for pos, i in enumerate(order):
                cid = chunks[i]["chunk_id"]
                best[cid][name] = float(s[i])
                if pos < 3:
                    top3[cid] += 1

    fused = rrf_fuse(rankings)
    fused_original = rrf_fuse(original_rankings)
    ranked_ids = sorted(fused, key=lambda cid: -fused[cid])
    by_id = {c["chunk_id"]: c for c in chunks}
    signals = {
        cid: {**best[cid], "rrf": fused_original[cid], "top3_frac": top3[cid] / len(original_rankings)}
        for cid in ranked_ids
    }
    signals["__rrf_sorted__"] = sorted(fused_original.values(), reverse=True)
    return [by_id[cid] for cid in ranked_ids[:min(pool, len(ranked_ids))]], signals


def hybrid_retrieve(question: str, pool: int = CANDIDATE_POOL, use_translation: bool = True,
                    teams: set[str] | None = None):
    return retrieve_with_signals(question, pool, use_translation, teams)[0]


def rerank(question: str, candidates, top_k: int = FINAL_K):
    """toy reranker (راجع تعليق مفصّل في مشروع RAG السابق): تغطية كلمات + تطابق
    عبارة. يُستخدم فقط عند غياب مفتاح API أو فشل llm_rerank — راجعها أدناه
    للترتيب الحقيقي القائم على الفهم لا تقاطع الكلمات."""
    q_tokens = set(_tokenize(question))
    scored = []
    for c in candidates:
        c_tokens_set = set(_tokenize(c["embed_text"]))
        coverage = len(q_tokens & c_tokens_set) / max(len(q_tokens), 1)
        phrase_bonus = 0.2 if normalize_arabic(question).strip("؟?") in normalize_arabic(c["embed_text"]) else 0.0
        scored.append((coverage + phrase_bonus, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


def llm_rerank(question: str, candidates: list, top_k: int = FINAL_K):
    """إعادة ترتيب حقيقية بحكم LLM (فهم المعنى) بدل تقاطع الكلمات — هذا هو
    الإصلاح المباشر للفجوة التي كشفها stage3_evaluation/hard_case_test.py:
    الـ reranker الساذج (rerank أعلاه) يتوه بسهولة في الأسئلة الغامضة أو
    المتقاطعة بين مواضيع لأنه لا يفهم المعنى، فقط يعدّ الكلمات المشتركة.

    مهمة "خفيفة" (رد قصير: أرقام فقط) — مُسنَدة أساسًا لـ Hugging Face، مع
    OpenRouter كمزوّد احتياطي يستلمها لو HF فشل. يعيد None عند غياب كل مزوّد
    أو فشل الاستدعاء أو تعذّر تفسير رد النموذج — عندها يستخدم المستدعي
    rerank() التقليدي كخط احتياطي آمن، دون كسر أي سلوك موجود."""
    if not candidates:
        return None

    blocks = []
    for i, c in enumerate(candidates, start=1):
        kind = "مقالة KB رسمية" if c["source_type"] == "kb" else "تذكرة سابقة"
        blocks.append(f"{i}. [{kind}] {c['title']}\n{c['text'][:200]}")
    prompt = (
        f"رسالة العميل: {question}\n\nالمصادر المرشحة:\n" + "\n\n".join(blocks) +
        "\n\nرتّب أرقام المصادر من الأكثر صلة فعليًا بمضمون رسالة العميل إلى الأقل. "
        "استبعد تمامًا أي رقم مصدره غير ذي صلة حقيقية بالسؤال. "
        "أعد الأرقام فقط مفصولة بفواصل بدون أي شرح إضافي، مثال: 3,1,4"
    )

    raw = None
    hf_client = get_hf_client()
    if hf_client is not None:
        try:
            response = hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], max_tokens=50,
            )
            raw = response.choices[0].message.content.strip()
        except Exception:
            logging.getLogger("agent_copilot").warning("فشل llm_rerank عبر Hugging Face، جارٍ تسليم المهمة لـ OpenRouter", exc_info=True)

    if raw is None:
        or_client = get_openrouter_client()
        if or_client is not None:
            try:
                response = or_client.chat.completions.create(
                    model=get_openrouter_model(),
                    messages=[{"role": "user", "content": prompt}], max_tokens=50,
                )
                raw = response.choices[0].message.content.strip()
            except Exception:
                logging.getLogger("agent_copilot").warning("فشل llm_rerank عبر OpenRouter", exc_info=True)

    if raw is None:
        return None

    seen_indices = []
    for tok in re.findall(r"\d+", raw):
        idx = int(tok) - 1
        if 0 <= idx < len(candidates) and idx not in seen_indices:
            seen_indices.append(idx)

    if not seen_indices:
        return None
    return [candidates[i] for i in seen_indices[:top_k]]


def retrieve(question: str, teams: set[str] | None = None):
    """ترتيب LLM إن توفر، وإلا ترتيب الدمج الهجين نفسه (أدق من rerank الساذج
    القائم على تقاطع الكلمات، خصوصًا مع رسائل العامية)."""
    candidates = hybrid_retrieve(question, teams=teams)
    return llm_rerank(question, candidates) or candidates[:FINAL_K]


def draft(question: str) -> str:
    retrieved = retrieve(question)
    return generate_draft(question, retrieved)


if __name__ == "__main__":
    build_index()
    q = sys.argv[1] if len(sys.argv) > 1 else "الكود بتاع السحب من الوكيل خلص عليا"
    print(f"\nرسالة العميل: {q}\n")
    candidates = hybrid_retrieve(q)
    print(f"مرشحو المرحلة الأولى (Hybrid, {len(candidates)}):")
    for c in candidates:
        print(f"  - [{c['source_type']}] {c['title']}")
    final = rerank(q, candidates)
    print(f"\nبعد Reranking (أفضل {len(final)}):")
    for c in final:
        print(f"  - [{c['source_type']}] {c['title']}")
    print("\nالمسودة:")
    print(draft(q))
