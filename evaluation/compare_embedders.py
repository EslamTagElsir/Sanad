"""
evaluation/compare_embedders.py — مقارنة نماذج الـ embedding على المجموعة الذهبية
--------------------------------------------------------------------------------
يشغّل evaluation.golden_eval مرة لكل نموذج (في عملية منفصلة حتى تتحرر ذاكرة
الـ GPU بين النماذج، وبفهرس مؤقت لكل نموذج فلا يُلمس الفهرس الحقيقي)، ويطبع
جدول مقارنة. "" = بدون نموذج dense (المؤشرات المحلية فقط).

    python -m evaluation.compare_embedders "" sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 Qwen/Qwen3-Embedding-0.6B
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

_CHILD = """
import json, sys, time
sys.path.insert(0, {root!r})
from pathlib import Path
from stage2_hybrid import rag
rag.STORE_DIR = Path({store!r}); rag.INDEX_PATH = rag.STORE_DIR / "index_v2.pkl"
t0 = time.time(); rag.build_index(); build_s = time.time() - t0
from evaluation.golden_eval import run
t0 = time.time(); report = run(); eval_s = time.time() - t0
report["timing"] = {{"build_s": build_s, "eval_s": eval_s}}
print("REPORT" + json.dumps(report))
"""


def evaluate(model: str) -> dict:
    env = {**os.environ, "AGENT_ASSIST_EMBED_MODEL": model, "OPENROUTER_API_KEY": "", "HF_API_TOKEN": "",
           "TRANSFORMERS_VERBOSITY": "error"}
    with tempfile.TemporaryDirectory() as store:
        out = subprocess.run([sys.executable, "-c", _CHILD.format(root=str(ROOT), store=store)],
                             env=env, capture_output=True, text=True, cwd=ROOT)
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT")), None)
    if line is None:
        raise RuntimeError(f"فشل تقييم {model or 'lexical'}:\n{out.stderr[-2000:]}")
    return json.loads(line[len("REPORT"):])


def main(models: list[str]):
    rows = []
    for m in models:
        r = evaluate(m)
        rows.append((m or "(بدون dense)", r))
    cols = [("hit@1", "retrieval", "hit@1"), ("hit@4", "retrieval", "hit@4"), ("mrr", "retrieval", "mrr"),
            ("auc", "calibration_cv", "auc"), ("ece", "calibration_cv", "ece"), ("brier", "calibration_cv", "brier"),
            ("oos_esc", "calibration_cv", "oos_escalation_rate"),
            ("false_esc", "calibration_cv", "correct_false_escalation_rate"),
            ("send_prec", "calibration_cv", "send_ready_precision"),
            ("eval_s", "timing", "eval_s")]
    print("\n" + "model".ljust(58) + "".join(c[0].rjust(10) for c in cols))
    for name, r in rows:
        print(name.ljust(58) + "".join(f"{r[s][k]:10.3f}" for _, s, k in cols))


if __name__ == "__main__":
    main(sys.argv[1:] or ["", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"])
