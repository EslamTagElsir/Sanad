"""
evaluation/golden_eval.py — تقييم الاسترجاع والثقة على المجموعة الذهبية
-----------------------------------------------------------------------
يقيس ثلاثة أشياء منفصلة (كلها بدون أي استدعاء LLM، حتميّة وقابلة للتكرار):

1. جودة الاسترجاع (الأسئلة داخل النطاق فقط): hit@1، hit@K، MRR — البحث في
   كل المصادر (كما يعمل /draft فعليًا).
2. جودة معايرة الثقة: Brier score، ECE، AUC — على احتمالات out-of-fold
   (cross-validation) حتى لا يُقاس النموذج على نفس الأسئلة التي تدرّب عليها.
3. أثر عتبات القرار: نسبة تصعيد الأسئلة خارج النطاق، نسبة التصعيد الخاطئ
   لأسئلة صحيحة، ودقة send_ready.

شغّله من داخل مجلد المشروع (sanad):
    python -m evaluation.golden_eval
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from stage2_hybrid.rag import retrieve_with_signals, FINAL_K
from stage2_hybrid.confidence import (
    load_golden, cross_validated_probs, is_correct_source, ESCALATE_BELOW, SEND_READY_ABOVE,
)


def _lang(text: str) -> str:
    import re
    return "ar" if len(re.findall(r"[؀-ۿ]", text)) >= len(re.findall(r"[A-Za-z]", text)) else "en"


def retrieval_metrics(items: list[dict]) -> dict:
    hits1, hitsk, rr, langs, misses = [], [], [], [], []
    for it in items:
        if not it["in_scope"]:
            continue
        candidates, _ = retrieve_with_signals(it["question"], use_translation=False)
        ranks = [i for i, c in enumerate(candidates) if is_correct_source(c, it["expected_sources"])]
        hits1.append(bool(ranks) and ranks[0] == 0)
        hitsk.append(bool(ranks) and ranks[0] < FINAL_K)
        rr.append(1.0 / (ranks[0] + 1) if ranks else 0.0)
        langs.append(_lang(it["question"]))
        if not hits1[-1]:
            misses.append(f"{it['id']} → {candidates[0]['source_id']} ({candidates[0]['title'][:50]})")
    report = {"hit@1": float(np.mean(hits1)), f"hit@{FINAL_K}": float(np.mean(hitsk)), "mrr": float(np.mean(rr)), "n": len(hits1)}
    for lang in ("ar", "en"):
        sel = [h for h, l in zip(hits1, langs) if l == lang]
        report[f"hit@1_{lang}"] = float(np.mean(sel)) if sel else float("nan")
        report[f"n_{lang}"] = len(sel)
    report["misses@1"] = misses
    return report


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def calibration_metrics(items: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score
    probs, y = cross_validated_probs(items)
    in_scope = np.array([it["in_scope"] for it in items])
    escalated = probs < ESCALATE_BELOW
    send_ready = probs > SEND_READY_ABOVE
    return {
        "brier": float(np.mean((probs - y) ** 2)),
        "ece": expected_calibration_error(probs, y),
        "auc": float(roc_auc_score(y, probs)),
        "oos_escalation_rate": float(escalated[~in_scope].mean()),
        "correct_false_escalation_rate": float(escalated[y == 1].mean()),
        "send_ready_precision": float(y[send_ready].mean()) if send_ready.any() else float("nan"),
        "send_ready_count": int(send_ready.sum()),
        "n": len(y),
    }


def run() -> dict:
    items = load_golden()
    return {
        "retrieval": retrieval_metrics(items),
        "calibration_cv": calibration_metrics(items),
    }


if __name__ == "__main__":
    for section, metrics in run().items():
        print(f"\n[{section}]")
        for k, v in metrics.items():
            if isinstance(v, list):
                print(f"  {k}:"); [print(f"    - {x}") for x in v]
            else:
                print(f"  {k:32s} {v:.3f}" if isinstance(v, float) else f"  {k:32s} {v}")
