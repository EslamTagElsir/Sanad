"""
offline_embeddings.py
----------------------
هذه البيئة التعليمية تعمل بلا اتصال بمزوّدي embedding خارجيين، فنستخدم TF-IDF
(عبر scikit-learn) كبديل محلي وحتمي. راجع ملاحظة مهمة في نهاية الملف عن حدود
هذا البديل مع بيانات ثنائية اللغة (عربي/إنجليزي مختلط) — وهي بالضبط حالتنا هنا.

في مشروعك الحقيقي استبدل OfflineEmbedder بنموذج embedding متعدد اللغات حقيقي
(Qwen3-Embedding مثلاً) بنفس الواجهة (fit_transform / transform).
"""

import re
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")


def normalize_arabic(text: str) -> str:
    """توحيد بسيط للعربية: إزالة التشكيل، توحيد الألف/الياء/الهمزات، وإزالة "ال"
    التعريف من بداية الكلمة. يرفع جودة مطابقة TF-IDF/BM25 بشكل ملموس دون تعقيد.
    النص الإنجليزي يمر دون تغيير."""
    text = _ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"(?<![؀-ۿ])ال(?=[؀-ۿ]{2,})", "", text)
    return text


# ---------------------------------------------------------------------------
# معجم مفاهيم ثنائي اللغة + لهجات: يربط صيغ العامية المصرية/السودانية والفصحى
# والإنجليزية لنفس المعنى بـ "رمز مفهوم" واحد (مثل __transfer__) يُلحَق بالنص.
# هذا يعالج الفجوة التي ظهرت في رسالة "حولت فلوس و الفلوس م وصلتش": لا توجد
# كلمة حرفية مشتركة مع مقالة "فشل التحويل ولم يتم خصم أو رد المبلغ" الفصحى،
# لكن الاثنين يشتركان في __transfer__ و __money__ و __not_arrived__.
# المفاتيح تُكتب بأي صيغة وتُوحَّد عبر normalize_arabic عند التحميل.
# ---------------------------------------------------------------------------
_CONCEPTS = {
    "transfer": "حولت حولتله حوّلت حول تحويل تحويلات يحول بعت بعتت بعتله ارسلت ارسال transfer transferred transfers sent send sending",
    "money": "فلوس الفلوس قروش مبلغ المبلغ جنيه جنيهات رصيد egp money amount balance funds",
    "not_arrived": "وصلتش وصلش وصلتو واصلاه واصلتش مواصلتش ماوصلتش يوصل وصلت يصل وصول استلم استلمش فشل failed fail arrive arrived received receive pending",
    "deducted": "اتخصم اتخصمت اتسحب نقص خصم مخصوم مخصومًا deducted debited taken",
    "refund": "يرجع ترجع رجعت رجوع استرجاع استرداد رد refund refunds reversal reversed back",
    "freeze": "اتجمد اتجمدت جمد يجمد تجميد unfreeze مجمد متجمد اتقفل اتوقف موقوف frozen freeze locked suspended blocked",
    "kyc": "kyc هويه هويتي بطاقه قومي سيلفي توثيق موثق وثقت id identity verification verify verified selfie",
    "otp": "otp رمز كود التحقق رساله رسايل sms code verification",
    "card": "فيزا كارت كارد بطاقه بنكيه ماستر card cards visa debit credit mastercard",
    "link": "اربط ربط اضيف اضافه link linking linked add remove",
    "fees": "رسوم عموله عمولات مصاريف اتحاسب fee fees charge charged commission pricing",
    "limit": "حد الحد حدود اقصى سقف ليمت limit limits capped cap maximum",
    "cashout": "سحب اسحب وكيل الوكيل وكلاء agent agents cashout cash-out withdraw withdrawal",
    "close": "اقفل اقفال اغلاق قفل الغي امسح close closing closure delete",
    "dispute": "معامله معاملات عملتهاش عملتها اعرفها نزاع احتيال سرقه dispute unauthorized fraud recognize",
    "business": "بيزنس اعمال تجاري محل شركه سجل business shop merchant commercial",
    "wrong_recipient": "غلط خطا بالغلط wrong mistake mistakenly",
}


def _concept_index() -> dict:
    index = {}
    for concept, words in _CONCEPTS.items():
        for w in words.split():
            for form in {w.lower(), normalize_arabic(w.lower())}:
                index.setdefault(form, set()).add(concept)
    return index


_CONCEPT_INDEX = _concept_index()
_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


# لواحق عامية تُنزع لمطابقة الجذر: ضمائر المفعول المتصلة (بعد نزع النفي "ش").
_SUFFIXES = ("هولي", "هوله", "هالي", "هاله", "ها", "هم", "هو", "له", "لي", "ني", "و", "ه")


def concepts_of(text: str) -> set:
    """رموز المفاهيم الموجودة في النص. نجرّب أيضًا نزع بادئات عامية شائعة
    (و/ف/ب + "ال") ولاحقة النفي "ش" والضمائر المتصلة حتى تُطابَق "والفلوس"
    و"وصلتش" و"حولتها" و"وصلتهاش"."""
    found = set()
    for raw in _WORD_RE.findall(normalize_arabic(text.lower())):
        stems = {raw}
        if len(raw) > 3 and raw[0] in "وفب":
            stems.add(normalize_arabic(raw[1:]))
        stems |= {w[:-1] for w in list(stems) if len(w) > 4 and w.endswith("ش")}
        candidates = set(stems)
        for w in stems:
            for suf in _SUFFIXES:
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    candidates.add(w[:-len(suf)])
        for c in candidates:
            found |= _CONCEPT_INDEX.get(c, set())
    return found


def expand_with_concepts(text: str) -> str:
    """النص الأصلي + رموز مفاهيمه، لتُستخدم كمدخل موحّد لكل مؤشرات البحث."""
    tags = " ".join(f"__{c}__" for c in sorted(concepts_of(text)))
    return f"{text} {tags}".strip()


def normalize_lower(text: str) -> str:
    return normalize_arabic(text).lower()


def normalize_for_index(text: str) -> str:
    return normalize_arabic(expand_with_concepts(text)).lower()


class OfflineEmbedder:
    def __init__(self, max_features: int = 4000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            preprocessor=normalize_for_index,
            token_pattern=r"(?u)\b\w\w+\b|__\w+__",
        )
        self._fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("يجب استدعاء fit_transform أولًا قبل transform")
        return self.vectorizer.transform(texts).toarray()

    def save(self, path: Path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path):
        import pickle
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class CharEmbedder(OfflineEmbedder):
    """TF-IDF على n-grams حرفية (داخل حدود الكلمة): يلتقط تقارب الصيغ الصرفية
    واللهجية ("حولت"/"تحويل"، "اتخصمت"/"خصم") التي لا يراها TF-IDF الكلمات."""

    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=max_features,
            preprocessor=normalize_lower,
            sublinear_tf=True,
        )
        self._fitted = False


class DenseEmbedder:
    """نموذج embedding دلالي متعدد اللغات حقيقي (اختياري). يُفعَّل فقط عند
    ضبط AGENT_ASSIST_EMBED_MODEL (مثال: intfloat/multilingual-e5-small) مع تثبيت
    sentence-transformers؛ بدونه يعمل النظام بالمؤشرات المحلية فقط."""

    def __init__(self, model_name: str):
        import logging
        import os
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        # AGENT_ASSIST_EMBED_DEVICE=cpu/cuda لفرض جهاز؛ الافتراضي تلقائي.
        device = os.environ.get("AGENT_ASSIST_EMBED_DEVICE") or None
        try:
            # من الكاش المحلي أولًا (بدون شبكة)؛ التحميل من الإنترنت فقط إن لم يوجد.
            self.model = SentenceTransformer(model_name, local_files_only=True, device=device)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        try:
            self.model.encode(["probe"])
        except Exception:
            # بعض بيئات الـ GPU (مثل كروت Blackwell بدون C compiler لـ triton)
            # تفشل وقت التشغيل الفعلي فقط — نرجع للـ CPU بدل إسقاط الاسترجاع.
            logging.getLogger("agent_copilot").warning(
                "فشل تشغيل %s على %s، التحويل إلى CPU", model_name, self.model.device, exc_info=True)
            self.model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
        self._e5 = "e5" in model_name.lower()
        # نماذج مثل Qwen3-Embedding تأتي بـ prompt مخصص للأسئلة داخل إعداداتها
        # (prompts["query"])؛ المستندات تُرمَّز بدون prompt.
        self._query_prompt = "query" if "query" in (getattr(self.model, "prompts", None) or {}) else None

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        kwargs = {}
        if self._e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        elif is_query and self._query_prompt:
            kwargs["prompt_name"] = self._query_prompt
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=16, **kwargs))


def get_dense_embedder():
    import logging
    import os
    name = os.environ.get("AGENT_ASSIST_EMBED_MODEL", "").strip()
    if not name:
        return None
    try:
        return DenseEmbedder(name)
    except Exception:
        logging.getLogger("agent_copilot").warning(
            "تعذّر تحميل نموذج الـ embedding %s، الاستمرار بالمؤشرات المحلية فقط", name, exc_info=True)
        return None


def cosine_topk(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int):
    sims = cosine_similarity(query_vec.reshape(1, -1), doc_matrix)[0]
    top_idx = np.argsort(-sims)[:k]
    return list(top_idx), sims


# ---------------------------------------------------------------------------
# ملاحظة مهمة: هذا الـ embedder LEXICAL محلي اللغة، وليس دلاليًا متعدد اللغات.
# معناها العملي: سؤال عربي لن "يفهم" أنه يقابل معنى مقالة إنجليزية إلا إذا
# شاركا كلمات حرفية (أرقام، أسماء منتجات، مصطلحات إنجليزية مستخدَمة داخل
# الجملة العربية مثل "KYC" أو "OTP"). هذه قاعدة معرفتنا هنا مختلطة اللغة
# عمدًا (بعض المقالات عربي، بعضها إنجليزي)، فستشهد هذه الفجوة بنفسك في
# التقييم بالمرحلة 3. المعالجة العملية في المرحلة 2: توسيع/ترجمة السؤال قبل
# الاسترجاع (راجع stage2_hybrid/rag.py: translate_query_for_retrieval)، وهي
# نفس الفكرة التي يحلها نموذج embedding حقيقي متعدد اللغات تلقائيًا.
# ---------------------------------------------------------------------------
