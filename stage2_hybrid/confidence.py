"""
stage2_hybrid/confidence.py — ثقة معايَرة (Calibrated Confidence)
------------------------------------------------------------------
بديل classify_confidence القديمة (نسبة الكلمات الحرفية المشتركة بين السؤال
وأول مصدر). تلك الطريقة كانت تعطي 0% لرسالة عامية صحيحة تمامًا مثل
"حولت فلوس و الفلوس م وصلتش" لمجرد أن المقالة الرسمية مكتوبة بالفصحى.

الآن الثقة = احتمال أن يكون المصدر المسترجَع هو فعلًا المصدر الصحيح، وتُحسب
بنموذج Logistic Regression (Platt scaling) على تشابه نموذج الـ embedding بين
السؤال والمصدر. جُرّبت إشارات إضافية (الهامش عن أقرب منافس، البروز z-score عن
بقية القاعدة) وكانت أسوأ في اكتشاف الأسئلة خارج النطاق: قاعدة المعرفة صغيرة،
فحتى سؤال عن الطقس "يبرز" نسبيًا عن بقية المقالات.
النموذج يُدرَّب على data/golden_set.json، فالرقم الناتج احتمال فعلي قابل
للقياس: من بين المسودات بثقة ~0.8، يجب أن يكون ~80% منها بمصدر صحيح.
جودة المعايرة (Brier / ECE) تُقاس بـ cross-validation في evaluation/golden_eval.py.
"""

import json
import logging
from pathlib import Path

import numpy as np

from stage2_hybrid.rag import retrieve_with_signals

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
TRANSLATIONS_PATH = Path(__file__).parent.parent / "data" / "golden_translations.json"
logger = logging.getLogger("sanad")

# عتبات القرار على الاحتمال المعايَر (لا على نسبة كلمات): أقل من ESCALATE_BELOW
# = الأرجح أن أول مصدر خطأ أو لا يوجد مصدر مناسب → تصعيد. أعلى من
# SEND_READY_ABOVE + مصدر KB رسمي → جاهز للإرسال بعد مراجعة سريعة.
# راجع evaluation/golden_eval.py لأثر العتبتين على المجموعة الذهبية.
ESCALATE_BELOW = 0.4
SEND_READY_ABOVE = 0.8

FEATURES = ["dense"]


def is_correct_source(chunk: dict, expected_sources: list[str]) -> bool:
    """المصدر صحيح إذا كان نفسه ضمن المتوقع، أو قطعة من دليل السياسات تشرح
    نفس السياسة (تذكر مرجع المقالة المتوقعة). قطع الدليل التي تجمع مراجع كثيرة
    (مثل جدول الأرقام المرجعية) لا تُحتسب بمرجع واحد فيها، لأنها ليست إجابة
    مركّزة على سؤال بعينه."""
    if chunk["source_id"] in expected_sources:
        return True
    refs = set(chunk.get("refs", []))
    return 0 < len(refs) <= 3 and bool(refs & set(expected_sources))


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["items"]


def load_golden_translations() -> dict[str, str | None]:
    """ترجمات LLM محفوظة لأسئلة المجموعة الذهبية (evaluation/translate_golden.py)."""
    if not TRANSLATIONS_PATH.exists():
        return {}
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_features(question: str, chunk: dict, signals: dict) -> list[float]:
    return [signals[chunk["chunk_id"]]["dense"]]


def _training_rows(items: list[dict]):
    """لكل سؤال ذهبي: إشارات أول نتيجة بنفس صيغة وقت التشغيل (الأصل + ترجمة
    LLM) لكن بترجمة محفوظة حتى يكون التدريب حتميًا وبلا استدعاءات، والتسمية = 1
    إذا كان أول مصدر ضمن expected_sources."""
    translations = load_golden_translations()
    X, y = [], []
    for it in items:
        candidates, signals = retrieve_with_signals(
            it["question"], use_translation=False, translation=translations.get(it["id"]))
        if not candidates:
            continue
        top = candidates[0]
        X.append(extract_features(it["question"], top, signals))
        y.append(int(it["in_scope"] and is_correct_source(top, it["expected_sources"])))
    return np.array(X), np.array(y)


def _new_estimator():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


class ConfidenceModel:
    def __init__(self):
        self.estimator = None
        self.features: list[str] = FEATURES

    def fit(self, items: list[dict]) -> "ConfidenceModel":
        X, y = _training_rows(items)
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
            # بدون مجموعة ذهبية: التشابه الخام (غير معايَر) كتقدير احتياطي.
            return float(min(max(x[0], 0.0), 1.0))
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
