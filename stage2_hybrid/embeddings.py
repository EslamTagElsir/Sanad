"""
stage2_hybrid/embeddings.py — نموذج الـ embedding الدلالي
----------------------------------------------------------
الاسترجاع كله قائم على نموذج embedding متعدد اللغات حقيقي (لا TF-IDF ولا
مطابقة كلمات): يفهم أن "حولت فلوس ومش واصلة" و"فشل التحويل" و"transfer failed"
نفس المعنى بدون أي قواميس يدوية.

النموذج يُحدَّد عبر AGENT_ASSIST_EMBED_MODEL (الافتراضي Qwen/Qwen3-Embedding-0.6B،
الأفضل على المجموعة الذهبية — راجع evaluation/compare_embedders.py).
AGENT_ASSIST_EMBED_DEVICE=cpu/cuda لفرض جهاز؛ الافتراضي تلقائي مع رجوع للـ CPU.
"""

import logging
import os
from typing import List

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
logger = logging.getLogger("agent_copilot")


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
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
            logger.warning("فشل تشغيل %s على %s، التحويل إلى CPU", model_name, self.model.device, exc_info=True)
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


_EMBEDDERS: dict[str, Embedder] = {}


def get_embedder() -> Embedder:
    """نسخة واحدة من النموذج لكل اسم (تحميله مكلف). فشل التحميل يرفع استثناء:
    بدون نموذج لا يوجد استرجاع أصلًا، فالأفضل أن تفشل الخدمة عند البدء بوضوح."""
    name = os.environ.get("AGENT_ASSIST_EMBED_MODEL", "").strip() or DEFAULT_MODEL
    if name not in _EMBEDDERS:
        _EMBEDDERS[name] = Embedder(name)
    return _EMBEDDERS[name]
