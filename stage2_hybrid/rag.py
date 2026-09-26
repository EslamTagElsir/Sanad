"""
المرحلة 2: استرجاع دلالي متعدد المصادر + Reranking + معالجة الفجوة اللغوية
---------------------------------------------------------------------------
الاسترجاع قائم بالكامل على النماذج (لا TF-IDF ولا BM25 ولا قواميس يدوية):

1. مصدران مدمجان في فهرس واحد: مقالات KB الرسمية + التذاكر السابقة المحلولة.
   الآن سؤال مثل "الكود بتاع السحب خلص عليا" يستطيع أن يجد تذكرة 1129 حتى لو لم
   توجد صياغة مطابقة في أي مقالة KB.
2. استرجاع دلالي بنموذج embedding متعدد اللغات (راجع stage2_hybrid/embeddings.py)
   يفهم العامية والفصحى والإنجليزية دون مطابقة كلمات حرفية.
3. ترجمة/توسيع السؤال عبر LLM قبل البحث (Query Translation)، فقط إذا توفر
   OPENROUTER_API_KEY: نطلب من النموذج ترجمة سؤال العميل للغة الأخرى، ثم نبحث
   بالنسختين معًا وندمج النتائج. هذا يعالج الفجوة اللغوية الحقيقية التي رصدناها
   (سؤال إنجليزي عن الرسوم لا يصل لمقالة عربية بنفس المعنى). بدون مفتاح API، هذه
   الخطوة تُتجاوز ويعتمد الاسترجاع على النموذج متعدد اللغات وحده.
4. ترتيب نهائي بحكم LLM (llm_rerank) إن توفر مزوّد.

شغّله من داخل مجلد المشروع (sanad):
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

from common import (
    load_kb, load_past_tickets, load_kb_large, load_past_tickets_large,
    build_kb_chunks, build_ticket_chunks, generate_draft,
    load_policy_manual_sections, build_manual_chunks,
    openrouter_chat,
)
from stage2_hybrid.embeddings import get_embedder

# SANAD_LARGE_DATA=1 يضيف مجموعة البيانات الصناعية الكبيرة (data/*_large.json)
# فوق البيانات الأصلية عند بناء الفهرس — لاختبار stage4 API على حجم بيانات أكبر
# بكثير دون المساس بالبيانات الأصلية أو كسر اختبارات pytest الحالية (المُعطَّل
# افتراضيًا، فالسلوك الافتراضي يبقى كما هو تمامًا).
USE_LARGE_DATA = os.environ.get("SANAD_LARGE_DATA", "").lower() in ("1", "true", "yes")

STORE_DIR = Path(__file__).parent / ("store_large" if USE_LARGE_DATA else "store")
INDEX_PATH = STORE_DIR / "index_dense.pkl"

CANDIDATE_POOL = 8
FINAL_K = 4

# الفهرس يُحمَّل مرة واحدة في الذاكرة (كان يُقرأ من القرص مع كل طلب).
_INDEX: dict | None = None


def _sources_fingerprint() -> str:
    """بصمة ملفات المصادر: تغيير أي ملف (KB، تذاكر، دليل السياسات) يعيد بناء الفهرس."""
    import hashlib
    from common import DATA_DIR, POLICY_MANUAL_PATH
    h = hashlib.sha256()
    for p in [DATA_DIR / "kb_articles.json", DATA_DIR / "past_tickets.json", POLICY_MANUAL_PATH]:
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def build_index():
    global _INDEX
    STORE_DIR.mkdir(exist_ok=True)
    kb = load_kb()
    tickets = load_past_tickets()
    if USE_LARGE_DATA:
        kb = kb + load_kb_large()
        tickets = tickets + load_past_tickets_large()
    chunks = build_kb_chunks(kb) + build_ticket_chunks(tickets) + build_manual_chunks(load_policy_manual_sections())
    embedder = get_embedder()
    index = {
        "chunks": chunks,
        "model": embedder.model_name,
        "sources": _sources_fingerprint(),
        "matrix": embedder.encode([c["embed_text"] for c in chunks]),
    }
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    _INDEX = index

    n_kb = sum(1 for c in chunks if c["source_type"] == "kb")
    n_ticket = sum(1 for c in chunks if c["source_type"] == "past_ticket")
    n_manual = sum(1 for c in chunks if c["source_type"] == "manual")
    mode = " [وضع البيانات الكبيرة مفعّل]" if USE_LARGE_DATA else ""
    print(f"تمت فهرسة {len(chunks)} قطعة ({n_kb} من KB + {n_ticket} من تذاكر سابقة + {n_manual} من دليل السياسات) بنموذج {embedder.model_name}.{mode}")
    return index


def load_index() -> dict:
    """الفهرس المحفوظ يُستخدم فقط إن كان مبنيًا بنفس النموذج المضبوط حاليًا
    (متجهات نموذجين مختلفين غير قابلة للمقارنة)؛ غير ذلك يُعاد البناء."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        if index.get("model") == get_embedder().model_name and index.get("sources") == _sources_fingerprint():
            _INDEX = index
            return index
    return build_index()


# كاش للترجمات الناجحة فقط (الفشل مؤقت ويجب إعادة المحاولة): نفس الرسالة لا تستهلك
# طلبًا جديدًا من حصة OpenRouter. في الذاكرة، ويُفرَّغ جزئيًا عند تجاوز الحد.
_TRANSLATION_CACHE: dict[str, str] = {}
_TRANSLATION_CACHE_MAX = 2000


def translate_query_for_retrieval(question: str) -> str | None:
    """ترجمة السؤال للغة الأخرى (عربي↔إنجليزي) عبر OpenRouter لتوسيع البحث عبر
    قاعدة معرفة ثنائية اللغة ولرفع فهم رسائل العامية. بدون مزوّد أو عند الفشل
    تُتجاوز (None) — لا يجب أن تُسقط طلب /draft."""
    key = question.strip()
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]
    prompt = (
        f"ترجم النص التالي إلى العربية إن كان إنجليزيًا، أو إلى الإنجليزية إن كان عربيًا. "
        f"أعد الترجمة فقط بدون أي شرح إضافي:\n\n{question}"
    )
    translation = openrouter_chat([{"role": "user", "content": prompt}], max_tokens=200, task="translate")
    if translation:
        if len(_TRANSLATION_CACHE) >= _TRANSLATION_CACHE_MAX:
            for old in list(_TRANSLATION_CACHE)[: _TRANSLATION_CACHE_MAX // 2]:
                del _TRANSLATION_CACHE[old]
        _TRANSLATION_CACHE[key] = translation
    return translation


def retrieve_with_signals(question: str, pool: int = CANDIDATE_POOL, use_translation: bool = True,
                          translation: str | None = None):
    """الاسترجاع الدلالي + درجات التشابه لكل مصدر (تُستخدم لحساب الثقة المعايَرة).

    الترتيب: بتشابه الرسالة الأصلية (الأدق في ترتيب المصادر على المجموعة الذهبية).
    الإشارة (dense): أعلى تشابه بين الرسالة الأصلية وترجمتها بالـ LLM. النموذج
    يفهم العامية أضعف من الفصحى/الإنجليزية ("حولت فلوس و الفلوس م وصلتش" تشابهها
    0.35 مع مصدرها الصحيح، أقل من بعض الأسئلة خارج النطاق)، والترجمة ترفعها
    لـ 0.53 وتفصل الإجابات الصحيحة عن خارج النطاق تمامًا. نموذج الثقة مُدرَّب على
    نفس الصيغة بترجمات محفوظة (data/golden_translations.json).

    translation: ترجمة جاهزة بدل استدعاء LLM (للتدريب/التقييم الحتمي). بدون
    ترجمة (لا مزوّد متاح) تُحسب الإشارة من الأصل فقط، فتكون الثقة أقل والقرار
    أميل للتصعيد — الاتجاه الآمن.

    يُرجع (candidates, signals) حيث signals[chunk_id] = {"dense": تشابه}."""
    index = load_index()
    chunks = index["chunks"]
    embedder = get_embedder()

    queries = [question]
    if translation is None and use_translation:
        translation = translate_query_for_retrieval(question)
    if translation:
        queries.append(translation)

    sims = index["matrix"] @ embedder.encode(queries, is_query=True).T  # (n_chunks, n_queries)
    order = np.argsort(-sims[:, 0])
    best = sims.max(axis=1)

    signals = {chunks[i]["chunk_id"]: {"dense": float(best[i])} for i in order}
    return [chunks[i] for i in order[:min(pool, len(chunks))]], signals


def hybrid_retrieve(question: str, pool: int = CANDIDATE_POOL, use_translation: bool = True):
    return retrieve_with_signals(question, pool, use_translation)[0]


def llm_rerank(question: str, candidates: list, top_k: int = FINAL_K):
    """إعادة ترتيب بحكم LLM (فهم المعنى) فوق ترتيب نموذج الـ embedding.

    عبر OpenRouter. يعيد None عند غياب المزوّد أو فشل الاستدعاء أو تعذّر تفسير
    رد النموذج — عندها يستخدم المستدعي ترتيب نموذج الـ embedding كما هو."""
    if not candidates:
        return None

    blocks = []
    for i, c in enumerate(candidates, start=1):
        kind = {"kb": "مقالة KB رسمية", "manual": "دليل السياسات"}.get(c["source_type"], "تذكرة سابقة")
        blocks.append(f"{i}. [{kind}] {c['title']}\n{c['text'][:200]}")
    prompt = (
        f"رسالة العميل: {question}\n\nالمصادر المرشحة:\n" + "\n\n".join(blocks) +
        "\n\nرتّب أرقام المصادر من الأكثر صلة فعليًا بمضمون رسالة العميل إلى الأقل. "
        "استبعد تمامًا أي رقم مصدره غير ذي صلة حقيقية بالسؤال. "
        "أعد الأرقام فقط مفصولة بفواصل بدون أي شرح إضافي، مثال: 3,1,4"
    )

    raw = openrouter_chat([{"role": "user", "content": prompt}], max_tokens=50, task="rerank")
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


def retrieve(question: str):
    """ترتيب LLM إن توفر، وإلا ترتيب نموذج الـ embedding."""
    candidates = hybrid_retrieve(question)
    return llm_rerank(question, candidates) or candidates[:FINAL_K]


def draft(question: str) -> str:
    retrieved = retrieve(question)
    return generate_draft(question, retrieved)


if __name__ == "__main__":
    build_index()
    q = sys.argv[1] if len(sys.argv) > 1 else "الكود بتاع السحب من الوكيل خلص عليا"
    print(f"\nرسالة العميل: {q}\n")
    candidates = hybrid_retrieve(q)
    print(f"مرشحو المرحلة الأولى (embedding, {len(candidates)}):")
    for c in candidates:
        print(f"  - [{c['source_type']}] {c['title']}")
    final = llm_rerank(q, candidates) or candidates[:FINAL_K]
    print(f"\nبعد Reranking (أفضل {len(final)}):")
    for c in final:
        print(f"  - [{c['source_type']}] {c['title']}")
    print("\nالمسودة:")
    print(draft(q))
