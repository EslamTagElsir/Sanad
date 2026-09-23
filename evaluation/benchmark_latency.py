"""
evaluation/benchmark_latency.py — قياس زمن كل جزء في المشروع
------------------------------------------------------------
1. البدء: استيراد المكتبات، تحميل نموذج الـ embedding، بناء الفهرس، تدريب الثقة.
2. الأجزاء المحلية (بدون LLM، مكررة N مرة → الوسيط وp95): ترميز سؤال بالـ
   embedding، الاسترجاع الكامل، حساب الثقة.
3. طلبات /draft كاملة عبر API بمزوّد OpenRouter الحقيقي، مقسومة لمراحلها
   (timings في رد الخدمة)، ومعها زمن وموديل كل طلب OpenRouter من اللوج.

    python -m evaluation.benchmark_latency            # كل شيء (يستهلك ~4 طلبات OpenRouter لكل رسالة)
    python -m evaluation.benchmark_latency --no-llm   # الأجزاء المحلية فقط
"""

import json
import logging
import statistics
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
sys.path.append(str(Path(__file__).parent.parent))

MESSAGES = [
    "الفلوس ال حولتها م وصلتش",
    "My account got frozen, how do I unfreeze it?",
    "التحويل للبنك بياخد عمولة قد ايه؟",
    "do you sell iPhones?",
]


class _OpenRouterLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("OPENROUTER |"):
            self.lines.append(msg)


def _stats(samples_ms):
    s = sorted(samples_ms)
    return {"median": statistics.median(s), "p95": s[max(0, int(round(0.95 * len(s))) - 1)], "n": len(s)}


def _bench(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return _stats(out)


def main(with_llm: bool):
    report = {}

    # ---------- 1) البدء ----------
    t = time.perf_counter()
    from stage2_hybrid import rag, confidence
    from stage2_hybrid.embeddings import get_embedder
    report["startup_import_ms"] = (time.perf_counter() - t) * 1000 + (t - T_START) * 1000

    t = time.perf_counter(); embedder = get_embedder()
    report["startup_model_load_ms"] = (time.perf_counter() - t) * 1000
    report["embedding_device"] = str(embedder.model.device)

    t = time.perf_counter(); rag.build_index()
    report["startup_index_build_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter(); confidence.reset_model(); model = confidence.get_model()
    report["startup_confidence_fit_ms"] = (time.perf_counter() - t) * 1000

    # ---------- 2) الأجزاء المحلية ----------
    q = MESSAGES[0]
    report["embed_one_query"] = _bench(lambda: embedder.encode([q], is_query=True), 30)
    report["retrieval_no_llm"] = _bench(lambda: rag.retrieve_with_signals(q, use_translation=False), 30)
    cands, sig = rag.retrieve_with_signals(q, use_translation=False)
    report["confidence_predict"] = _bench(lambda: model.predict(q, cands[0], sig), 200)

    # ---------- 3) /draft كامل ----------
    if with_llm:
        from fastapi.testclient import TestClient
        from stage4_production import service, auth
        handler = _OpenRouterLog()
        logging.getLogger("agent_copilot").addHandler(handler)
        drafts = []
        with TestClient(service.app) as c:
            user = next(u for u in auth.load_users() if u["role"] == "employee")
            h = {"Authorization": "Bearer " + auth.create_access_token(user)}
            for msg in MESSAGES:
                before = len(handler.lines)
                t = time.perf_counter()
                body = c.post("/draft", json={"customer_message": msg}, headers=h).json()
                drafts.append({
                    "message": msg, "action": body["action"], "confidence": body["confidence"],
                    "total_ms": (time.perf_counter() - t) * 1000, "timings": body["timings"],
                    "openrouter": handler.lines[before:],
                })
        report["drafts"] = drafts

    _print(report)
    out = Path(__file__).parent / "latency_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(التقرير الكامل: {out})")


def _print(r):
    print("\n=== البدء (مرة واحدة عند تشغيل السيرفر) ===")
    for k, label in [("startup_import_ms", "استيراد المكتبات (torch, sentence-transformers...)"),
                     ("startup_model_load_ms", f"تحميل نموذج الـ embedding ({r['embedding_device']})"),
                     ("startup_index_build_ms", "بناء الفهرس (ترميز كل المصادر)"),
                     ("startup_confidence_fit_ms", "تدريب نموذج الثقة (65 سؤال ذهبي)")]:
        print(f"  {label:55s} {r[k]:9.0f} ms")
    print("\n=== الأجزاء المحلية لكل طلب (وسيط / p95) ===")
    for k, label in [("embed_one_query", "ترميز سؤال واحد بالـ embedding"),
                     ("retrieval_no_llm", "الاسترجاع الكامل (ترميز + تشابه + ترتيب)"),
                     ("confidence_predict", "حساب الثقة")]:
        print(f"  {label:55s} {r[k]['median']:8.1f} / {r[k]['p95']:8.1f} ms")
    for d in r.get("drafts", []):
        print(f"\n=== /draft: {d['message']} → {d['action']} ({d['confidence']:.2f}) — الإجمالي {d['total_ms']:.0f} ms ===")
        for k, v in d["timings"].items():
            print(f"  {k:14s} {v:9.0f} ms")
        for line in d["openrouter"]:
            print(f"    {line}")


if __name__ == "__main__":
    main(with_llm="--no-llm" not in sys.argv)
