"""المجموعة الذهبية: سلامة البيانات + حدود جودة دنيا للاسترجاع والثقة المعايَرة.
الحدود أدناه أقل قليلًا من القيم المقاسة فعليًا (راجع python -m evaluation.golden_eval)
حتى تكشف أي تراجع حقيقي دون أن تتذبذب مع تغييرات صغيرة في البيانات."""

import pytest

from common import load_kb, load_past_tickets
from evaluation.golden_eval import run, expected_calibration_error
from stage2_hybrid import confidence, rag
from stage2_hybrid.confidence import ESCALATE_BELOW, load_golden
from stage2_hybrid.rag import retrieve_with_signals
# الدالة الأصلية قبل أن يستبدلها conftest بالترجمات المحفوظة
from stage2_hybrid.rag import translate_query_for_retrieval as _original_translate
from stage4_production import service
from stage4_production.service import decide_action
from evaluation.tests.conftest import bearer

import numpy as np

SCREENSHOT_MESSAGE = "حولت فلوس و الفلوس م وصلتش"


@pytest.fixture(scope="module")
def report():
    return run()


# ---------- سلامة المجموعة الذهبية ----------

def test_golden_set_is_well_formed():
    items = load_golden()
    known = {a["id"]: a["team"] for a in load_kb()} | {t["id"]: t["team"] for t in load_past_tickets()}
    assert len({it["id"] for it in items}) == len(items)
    assert sum(it["in_scope"] for it in items) >= 40 and sum(not it["in_scope"] for it in items) >= 10
    for it in items:
        if it["in_scope"]:
            assert it["expected_sources"], it["id"]
            assert set(it["expected_sources"]) <= set(known), it["id"]
        else:
            assert it["expected_sources"] == []


def test_golden_questions_are_not_copies_of_indexed_tickets():
    ticket_msgs = {t["customer_message"].strip() for t in load_past_tickets()}
    assert not [it["id"] for it in load_golden() if it["question"].strip() in ticket_msgs]


# ---------- حدود الجودة ----------

def test_retrieval_quality(report):
    r = report["retrieval"]
    assert r["hit@4"] >= 0.80
    assert r["hit@1"] >= 0.60
    assert r["mrr"] >= 0.70


def test_confidence_is_calibrated_out_of_fold(report):
    c = report["calibration_cv"]
    assert c["auc"] >= 0.80
    assert c["ece"] <= 0.15
    assert c["brier"] <= 0.18


def test_decision_thresholds(report):
    c = report["calibration_cv"]
    assert c["oos_escalation_rate"] >= 0.85
    assert c["correct_false_escalation_rate"] <= 0.20
    assert c["send_ready_precision"] >= 0.85


def test_ece_helper():
    assert expected_calibration_error(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)


# ---------- حالات محددة ----------

def test_screenshot_case_is_no_longer_zero_confidence():
    """الحالة التي ظهرت في الواجهة بثقة 0% وتصعيد، رغم وجود مقالة وسابقة مطابقتين."""
    candidates, signals = retrieve_with_signals(SCREENSHOT_MESSAGE)  # ترجمة محفوظة بدل LLM (conftest)
    # المصدر الأول قد يكون مقالة/سابقة أو قطعة من دليل السياسات تشرح نفس السياسة
    assert confidence.is_correct_source(candidates[0], ["kb_failed_transfer", "ticket_1042", "kb_refund_policy"])
    # نفس حساب الخدمة: أعلى ثقة بين المصادر النهائية المعروضة (service.handle_request)
    conf = max(confidence.get_model().predict(SCREENSHOT_MESSAGE, c, signals) for c in candidates[:rag.FINAL_K])
    assert conf >= ESCALATE_BELOW
    assert decide_action(conf, candidates[0]) != "escalate"


def test_screenshot_case_through_api(client):
    body = client.post("/draft", json={"customer_message": SCREENSHOT_MESSAGE},
                       headers=bearer("sara.ahmed")).json()
    assert body["action"] in ("needs_review", "send_ready")
    assert body["citations"]


@pytest.mark.parametrize("q", ["do you sell iPhones?", "ايه رأيك في ماتش الاهلي امبارح"])
def test_out_of_scope_escalates_through_api(client, q):
    body = client.post("/draft", json={"customer_message": q}, headers=bearer("sara.ahmed")).json()
    assert body["action"] == "escalate"
    assert body["draft"] == "" and body["reason"]   # لا مسودة، مع سبب واضح للموظف


def test_confidence_is_a_probability():
    for it in load_golden()[:10]:
        candidates, signals = retrieve_with_signals(it["question"], use_translation=False)
        p = confidence.get_model().predict(it["question"], candidates[0], signals)
        assert 0.0 <= p <= 1.0
    assert confidence.get_model().predict("x", None, {}) == 0.0


def test_llm_rerank_reordering_does_not_collapse_confidence(client, monkeypatch):
    """ترتيب LLM قد يقدّم مصدرًا غير الأول هجينيًا؛ الثقة يجب ألا تنهار بسبب ذلك
    (كانت 0.24 → تصعيد خاطئ لنفس رسالة لقطة الشاشة)."""
    monkeypatch.setattr(rag, "llm_rerank", lambda q, cands: list(reversed(cands[:4])))
    body = client.post("/draft", json={"customer_message": SCREENSHOT_MESSAGE},
                       headers=bearer("sara.ahmed")).json()
    assert body["action"] != "escalate"


SECOND_SCREENSHOT_MESSAGE = "الفلوس ال حولتها م وصلتش"


def test_second_screenshot_case_through_api(client):
    body = client.post("/draft", json={"customer_message": SECOND_SCREENSHOT_MESSAGE},
                       headers=bearer("sara.ahmed")).json()
    assert body["action"] in ("needs_review", "send_ready"), body


def test_draft_request_ignores_legacy_queue_field(client):
    # واجهات قديمة ممكن لسه تبعت queue — يتجاهل ويبحث في كل المصادر.
    resp = client.post("/draft", json={"customer_message": SCREENSHOT_MESSAGE, "queue": "security"},
                       headers=bearer("sara.ahmed"))
    assert resp.status_code == 200 and resp.json()["citations"]


def test_missing_translation_is_the_conservative_direction():
    """بدون مزوّد LLM (لا ترجمة) الثقة يجب أن تنخفض أو تبقى، لا أن ترتفع."""
    for q in [SCREENSHOT_MESSAGE, SECOND_SCREENSHOT_MESSAGE, "do you sell iPhones?"]:
        with_tr, s1 = retrieve_with_signals(q)
        without_tr, s2 = retrieve_with_signals(q, use_translation=False)
        m = confidence.get_model()
        assert m.predict(q, without_tr[0], s2) <= m.predict(q, with_tr[0], s1) + 1e-9


def test_empty_llm_draft_falls_back_instead_of_500(monkeypatch):
    """OpenRouter رجّع content=None للمسودة نفسها → كانت 500 في /draft."""
    import common
    from types import SimpleNamespace as NS
    empty = NS(model="x", choices=[NS(message=NS(content=None))])
    fake = NS(chat=NS(completions=NS(create=lambda **kw: empty)))
    monkeypatch.setattr(common, "get_openrouter_client", lambda: fake)
    chunk = {"source_type": "kb", "title": "t", "text": "نص"}
    out = common.generate_draft("سؤال", [chunk])
    assert isinstance(out, str) and out.strip()


def test_openrouter_call_disables_reasoning_and_sends_fallback_models(monkeypatch):
    import common
    from types import SimpleNamespace as NS
    seen = {}

    def create(**kw):
        seen.update(kw)
        return NS(model=kw["model"], choices=[NS(message=NS(content=" نعم "))])

    monkeypatch.setattr(common, "get_openrouter_client", lambda: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setenv("OPENROUTER_MODELS", "a/one, b/two")
    assert common.openrouter_chat([{"role": "user", "content": "x"}], 5, "t") == "نعم"
    assert seen["model"] == "a/one"
    assert seen["extra_body"] == {"models": ["a/one", "b/two"], "reasoning": {"enabled": False}}


def test_openrouter_retries_when_reasoning_is_mandatory(monkeypatch):
    """openrouter/free قد يختار موديلًا يرفض إيقاف التفكير (400) — نعيد بأقل تفكير."""
    import common
    import httpx
    import openai
    from types import SimpleNamespace as NS
    calls = []

    def create(**kw):
        calls.append(kw["extra_body"]["reasoning"])
        if kw["extra_body"]["reasoning"] == {"enabled": False}:
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise openai.BadRequestError("Reasoning is mandatory for this endpoint and cannot be disabled.",
                                         response=httpx.Response(400, request=req), body=None)
        return NS(model="m", choices=[NS(message=NS(content="ok"))])

    monkeypatch.setattr(common, "get_openrouter_client", lambda: NS(chat=NS(completions=NS(create=create))))
    assert common.openrouter_chat([{"role": "user", "content": "x"}], 5, "t") == "ok"
    assert calls == [{"enabled": False}, {"effort": "low", "exclude": True}]


def test_daily_quota_exhaustion_short_circuits_later_calls(monkeypatch):
    """بعد 429 "free-models-per-day" لا نرسل طلبات (كانت تأخذ حتى 40s للرفض)."""
    import time
    import common
    import httpx
    import openai
    from types import SimpleNamespace as NS
    calls = []
    reset_ms = int((time.time() + 600) * 1000)

    def create(**kw):
        calls.append(1)
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        raise openai.RateLimitError(
            f"Rate limit exceeded: free-models-per-day. {{'X-RateLimit-Reset': '{reset_ms}'}}",
            response=httpx.Response(429, request=req), body=None)

    monkeypatch.setattr(common, "_quota_blocked_until", 0.0)
    monkeypatch.setattr(common, "get_openrouter_client", lambda: NS(chat=NS(completions=NS(create=create))))
    assert common.openrouter_chat([{"role": "user", "content": "x"}], 5, "t") is None
    assert common._quota_blocked_until == pytest.approx(reset_ms / 1000)
    assert common.openrouter_chat([{"role": "user", "content": "x"}], 5, "t") is None
    assert len(calls) == 1


def test_policy_manual_pdf_is_indexed_with_clean_arabic():
    """دليل السياسات يُستخرج من الـ PDF: فصول بأسماء صحيحة، ونص عربي بالترتيب
    المنطقي (لا "خلال%90 في")، ومراجع كل قسم لمقالته."""
    manual = [c for c in rag.load_index()["chunks"] if c["source_type"] == "manual"]
    assert len(manual) > 50
    chapters = {c["title"].split(" — ")[0] for c in manual}
    assert "5. الرسوم والعمولات" in chapters and len(chapters) == 22
    policy = next(c for c in manual if c["title"].endswith("6.2 السياسة المعتمدة"))
    assert "في 90% من الحالات يرد المبلغ تلقائيا خلال ساعتين" in policy["text"]
    assert policy["refs"] == ["kb_failed_transfer", "kb_refund_policy"]
    frozen = next(c for c in manual if "18.5 حساب مجمد" in c["title"])
    assert frozen["refs"] == ["kb_account_freeze"]


def test_escalation_shows_sources_and_translation_reason(client, monkeypatch):
    """لقطة الشاشة: الترجمة فشلت فانخفضت الثقة وصُعّدت الحالة، والواجهة قالت
    "لا يوجد مصدر مناسب" رغم أن المصادر الصحيحة استُرجعت. الآن: لا مسودة، لكن
    المصادر تظهر مع سبب يذكر فشل الترجمة، والنتيجة لا تُخزَّن في الكاش."""
    from stage4_production import service
    msg = "الفلوس م جاتش علي الحساب بتاعي"
    monkeypatch.setattr(rag, "translate_query_for_retrieval", lambda q: None)
    monkeypatch.setattr(service.common, "get_openrouter_client", lambda: object())  # مزوّد مُهيَّأ لكنه فشل
    body = client.post("/draft", json={"customer_message": msg}, headers=bearer("sara.ahmed")).json()
    assert body["action"] == "escalate" and body["draft"] == ""
    assert body["citations"], "المصادر يجب أن تظهر للموظف حتى عند التصعيد"
    assert "الترجمة" in body["reason"]
    again = client.post("/draft", json={"customer_message": msg}, headers=bearer("sara.ahmed")).json()
    assert again["cached"] is False   # فشل مؤقت لا يُخزَّن


def test_translation_cache_avoids_repeat_llm_calls(monkeypatch):
    """نفس الرسالة لا تستهلك طلب ترجمة جديدًا، والفشل لا يُخزَّن."""
    calls = []
    answers = iter([None, "The money did not arrive"])   # فشل ثم نجاح

    def fake_chat(messages, max_tokens, task):
        calls.append(task)
        return next(answers)

    monkeypatch.setattr(rag, "openrouter_chat", fake_chat)
    monkeypatch.setattr(rag, "_TRANSLATION_CACHE", {})
    assert _original_translate("رسالة") is None                      # فشل: لا يُخزَّن
    assert _original_translate("رسالة") == "The money did not arrive"  # إعادة محاولة
    assert _original_translate("رسالة") == "The money did not arrive"  # من الكاش
    assert calls == ["translate", "translate"]
