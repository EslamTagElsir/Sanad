"""
evaluation/generation_eval.py — تقييم التوليد (المسودات) end-to-end
-------------------------------------------------------------------
يكمّل evaluation/golden_eval.py (الاسترجاع والثقة، بدون LLM) بتقييم المسودة
النهائية التي يراها الموظف، عبر handle_request الحقيقي ومزوّد OpenRouter الحقيقي.

المقاييس (كلها آلية وحتمية على نص المسودة — بدون LLM حَكَم، لتوفير الحصة):
  - key_fact_recall: هل ذكرت المسودة الحقائق الجوهرية للسياسة (مثلًا "4 ساعات" و
    "يومي عمل" لتحويل فاشل)؟ كل مجموعة بدائل يجب أن يتحقق منها واحد على الأقل.
  - unsupported_numbers: أرقام في المسودة غير موجودة في المصادر المستشهد بها ولا
    في رسالة العميل (مؤشر مباشر على اختلاق الأرقام).
  - language_match: الرد بلغة رسالة العميل (عربي/إنجليزي).
  - citation_correct: أحد المصادر المعروضة صحيح للسؤال (is_correct_source).
  - action: للأسئلة خارج النطاق يجب "escalate"، وللأسئلة داخل النطاق يجب ألا تُصعَّد.

لتوفير حصة OpenRouter (المجانية 50 طلب/يوم): الترجمة من الترجمات المحفوظة
للمجموعة الذهبية، وترتيب الـ LLM معطّل افتراضيًا (--with-rerank لتفعيله)،
فالتكلفة ≈ طلب واحد لكل مسودة.

    python -m evaluation.generation_eval
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# سؤال واحد صعب لكل موضوع (عامية غالبًا) + الحقائق الجوهرية المتوقعة في المسودة.
# كل عنصر في key_facts = مجموعة بدائل (regex)، ويجب أن تتحقق كل مجموعة.
CASES = {
    "g_fail_5": [r"4\s*(ساعات|hours)|أربع ساعات", r"يوم(ي|ين) عمل|business days|48"],
    "g_kyc_1": [r"24\s*(ساعة|hours)"],
    "g_lim_1": [r"5[,،]?000|60[,،]?000"],
    "g_ref_1": [r"تواصل|contact", r"لا (يمكن|نستطيع|يمكننا)|cannot|can't|unable"],
    "g_frz_1": [r"خط الأمان|security line|الأمان", r"15\s*دقيقة|ساعة|hour"],
    "g_otp_1": [r"90", r"مكالمة|voice|call"],
    "g_card_4": [r"1\s*(جنيه|EGP)|EGP\s*1|one pound", r"(يُ?رد|refund|تلقائي|automatic)"],
    "g_fee_1": [r"1\s*[%٪]", r"50|5\s*جنيه"],
    "g_cash_4": [r"(كود|رمز|code).{0,40}(جديد|new)|(جديد|new).{0,40}(كود|رمز|code)", r"(لا|not|no).{0,30}(يخصم|خصم|deduct)"],
    "g_disp_1": [r"10\s*(أيام|days)", r"نموذج|form"],
    "g_close_1": [r"صفر|zero", r"5\s*(أيام|days)"],
    "g_biz_1": [r"سجل\S*\s*(ال)?تجاري|commercial registration", r"جديد|منفصل|separate|new"],
}
OOS_CASES = ["g_oos_1", "g_oos_2", "g_oos_9", "g_oos_16"]

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _numbers(text: str) -> set[str]:
    text = text.translate(_ARABIC_DIGITS)
    # تجاهل ترقيم القوائم في بداية السطر ("1)"، "2."، "- 3.")
    text = re.sub(r"(?m)^\s*[-*•]?\s*\d{1,2}\s*[).:-]\s", " ", text)
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def _is_arabic(text: str) -> bool:
    return len(re.findall(r"[؀-ۿ]", text)) >= len(re.findall(r"[A-Za-z]", text))


def main(with_rerank: bool):
    from stage2_hybrid import rag, confidence
    from stage4_production import service

    items = {it["id"]: it for it in confidence.load_golden()}
    translations = confidence.load_golden_translations()
    by_question = {it["question"]: translations.get(i) for i, it in items.items()}
    rag.translate_query_for_retrieval = lambda q: by_question.get(q)
    if not with_rerank:
        rag.llm_rerank = lambda q, cands: None

    rag.load_index()
    confidence.get_model()
    chunk_by_title = {}
    for c in rag.load_index()["chunks"]:
        chunk_by_title.setdefault(c["title"], []).append(c)

    rows = []
    for cid in list(CASES) + OOS_CASES:
        it = items[cid]
        t = time.perf_counter()
        resp = service.handle_request(it["question"])
        latency = time.perf_counter() - t
        cited = [c for cit in resp.citations for c in chunk_by_title.get(cit.title, [])]
        # السياق كما يراه الـ LLM في build_prompt: عنوان المصدر + نصه (العناوين فيها أرقام أقسام الدليل)
        context = "\n".join(f"{c['title']}\n{c['text']}" for c in cited) + "\n" + it["question"]
        draft = resp.draft
        generation_failed = draft.startswith("[وضع بدون مفتاح API")
        row = {"id": cid, "question": it["question"], "in_scope": it["in_scope"], "action": resp.action,
               "confidence": resp.confidence, "latency_s": round(latency, 1), "generation_failed": generation_failed,
               "draft": draft, "citations": [c.title for c in resp.citations]}
        if it["in_scope"]:
            facts = [bool(re.search(p, draft.translate(_ARABIC_DIGITS), re.I | re.S)) for p in CASES[cid]]
            unsupported = sorted(_numbers(draft) - _numbers(context)) if resp.action != "escalate" else []
            row.update({
                "key_facts": facts,
                "unsupported_numbers": unsupported,
                "language_match": _is_arabic(draft) == _is_arabic(it["question"]),
                "citation_correct": any(confidence.is_correct_source(c, it["expected_sources"]) for c in cited),
            })
        rows.append(row)
        print(f"  {cid:10s} {resp.action:13s} conf={resp.confidence:.2f} {latency:5.1f}s"
              + ("" if not it["in_scope"] else f" facts={sum(row['key_facts'])}/{len(row['key_facts'])}"
                 f" unsupported={row['unsupported_numbers']}{' GEN-FAILED' if generation_failed else ''}"), flush=True)

    ins = [r for r in rows if r["in_scope"]]
    answered = [r for r in ins if r["action"] != "escalate"]
    generated = [r for r in answered if not r["generation_failed"]]
    summary = {
        "in_scope_n": len(ins),
        "in_scope_not_escalated": len(answered) / len(ins),
        "generation_failed": sum(r["generation_failed"] for r in answered),
        "key_fact_recall": (sum(sum(r["key_facts"]) for r in generated) / max(1, sum(len(r["key_facts"]) for r in generated))),
        "drafts_with_all_key_facts": sum(all(r["key_facts"]) for r in generated) / max(1, len(generated)),
        "drafts_with_unsupported_numbers": sum(bool(r["unsupported_numbers"]) for r in generated) / max(1, len(generated)),
        "language_match": sum(r["language_match"] for r in generated) / max(1, len(generated)),
        "citation_correct": sum(r["citation_correct"] for r in answered) / max(1, len(answered)),
        "oos_escalated": sum(r["action"] == "escalate" for r in rows if not r["in_scope"]) / len(OOS_CASES),
        "median_latency_s": sorted(r["latency_s"] for r in answered)[len(answered) // 2] if answered else None,
    }
    print("\n[generation]")
    for k, v in summary.items():
        print(f"  {k:32s} {v:.3f}" if isinstance(v, float) else f"  {k:32s} {v}")
    out = Path(__file__).parent / "generation_report.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(المسودات الكاملة: {out})")


if __name__ == "__main__":
    main(with_rerank="--with-rerank" in sys.argv)
