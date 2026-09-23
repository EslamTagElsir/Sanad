"""
stage2_hybrid/confidence.py — ثقة معايَرة (Calibrated Confidence)
------------------------------------------------------------------
بديل classify_confidence القديمة (نسبة الكلمات الحرفية المشتركة بين السؤال
وأول مصدر). تلك الطريقة كانت تعطي 0% لرسالة عامية صحيحة تمامًا مثل
"حولت فلوس و الفلوس م وصلتش" لمجرد أن المقالة الرسمية مكتوبة بالفصحى.

الآن الثقة = احتمال أن يكون أول مصدر مسترجَع هو فعلًا المصدر الصحيح، وتُحسب
بنموذج Logistic Regression على إشارات الاسترجاع المتعددة (تشابه دلالي، تشابه
حروف، تشابه كلمات، BM25، تغطية المفاهيم، اتفاق المؤشرات، هامش الترتيب).
النموذج يُدرَّب على data/golden_set.json، فالرقم الناتج احتمال فعلي قابل
للقياس: من بين المسودات بثقة ~0.8، يجب أن يكون ~80% منها بمصدر صحيح.
جودة المعايرة (Brier / ECE) تُقاس بـ cross-validation في evaluation/golden_eval.py.
"""

import json
import logging
from pathlib import Path

import numpy as np

from stage2_hybrid.rag import retrieve_with_signals, load_index

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
logger = logging.getLogger("agent_copilot")

# عتبات القرار على الاحتمال المعايَر (لا على نسبة كلمات): أقل من ESCALATE_BELOW
# = الأرجح أن أول مصدر خطأ أو لا يوجد مصدر مناسب → تصعيد. أعلى من
# SEND_READY_ABOVE + مصدر KB رسمي → جاهز للإرسال بعد مراجعة سريعة.
# راجع evaluation/golden_eval.py لأثر العتبتين على المجموعة الذهبية.
ESCALATE_BELOW = 0.4
SEND_READY_ABOVE = 0.8

BASE_FEATURES = ["char", "word", "bm25_sq", "concept", "top3_frac", "rrf_margin"]


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["items"]


def feature_names() -> list[str]:
    return (["dense"] if load_index().get("dense") is not None else []) + BASE_FEATURES


def extract_features(question: str, chunk: dict, signals: dict) -> list[float]:
    s = signals[chunk["chunk_id"]]
    others = [v["rrf"] for cid, v in signals.items() if cid != chunk["chunk_id"] and not cid.startswith("__")]
    # الهامش بوحدات "مرتبة واحدة" في RRF (1/61) — موجب إذا تصدّر المصدر بوضوح.
    rrf_margin = (s["rrf"] - max(others, default=0.0)) * 61
    values = {
        "dense": s["dense"],
        "char": s["char"],
        "word": s["word"],
        "bm25_sq": s["bm25"] / (s["bm25"] + 5.0),
        "concept": s["concept"],
        "top3_frac": s["top3_frac"],
        "rrf_margin": rrf_margin,
    }
    return [values[n] for n in feature_names()]


def _training_rows(items: list[dict]):
    """لكل سؤال ذهبي: إشارات أول نتيجة (بدون ترجمة LLM حتى يكون التدريب حتميًا)،
    والتسمية = 1 إذا كان أول مصدر ضمن expected_sources."""
    X, y = [], []
    for it in items:
        candidates, signals = retrieve_with_signals(it["question"], use_translation=False)
        if not candidates:
            continue
        top = candidates[0]
        X.append(extract_features(it["question"], top, signals))
        y.append(int(it["in_scope"] and top["source_id"] in it["expected_sources"]))
    return np.array(X), np.array(y)


def _new_estimator():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


class ConfidenceModel:
    def __init__(self):
        self.estimator = None
        self.features: list[str] = []

    def fit(self, items: list[dict]) -> "ConfidenceModel":
        X, y = _training_rows(items)
        self.features = feature_names()
        if len(set(y)) < 2:
            logger.warning("المجموعة الذهبية لا تحتوي الفئتين، الثقة ستكون غير معايَرة")
            self.estimator = None
            return self
        self.estimator = _new_estimator().fit(X, y)
        logger.info(f"نموذج الثقة مُدرَّب على {len(y)} مثالًا ({int(y.sum())} صحيح) | features={self.features}")
        return self

    def predict(self, question: str, chunk: dict | None, signals: dict) -> float:
        if chunk is None or chunk["chunk_id"] not in signals:
            return 0.0
        x = extract_features(question, chunk, signals)
        if self.estimator is None:
            # بدون مجموعة ذهبية: تقدير متحفظ غير معايَر (اتفاق المؤشرات × تغطية المفاهيم).
            f = dict(zip(feature_names(), x))
            return float(f["top3_frac"] * max(f["concept"], f["char"]))
        return float(self.estimator.predict_proba(np.array([x]))[0, 1])


_MODEL: ConfidenceModel | None = None


def get_model() -> ConfidenceModel:
    global _MODEL
    if _MODEL is None:
        items = load_golden() if GOLDEN_PATH.exists() else []
        _MODEL = ConfidenceModel().fit(items)
    return _MODEL


def reset_model():
    global _MODEL
    _MODEL = None


def cross_validated_probs(items: list[dict], folds: int = 5, seed: int = 0):
    """احتمالات out-of-fold لكل سؤال ذهبي (كل احتمال من نموذج لم يرَ ذلك السؤال)
    — هذا ما يُقاس عليه Brier/ECE حتى لا تكون المعايرة مُقاسة على بيانات التدريب."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X, y = _training_rows(items)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probs = cross_val_predict(_new_estimator(), X, y, cv=cv, method="predict_proba")[:, 1]
    return probs, y
