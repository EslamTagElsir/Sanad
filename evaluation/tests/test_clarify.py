"""الأسئلة التوضيحية للرسائل الغامضة + رد العميل على نفس التذكرة (SQLite)."""

import pytest

import common
from stage4_production import service, ticket_store
from evaluation.tests.conftest import bearer

VAGUE = "ليه قافلين الاكونت بتاعي؟ انا معملتش حاجة"
QUESTIONS = ["هل بتقصد إن الحساب اتجمد ومش قادر تدخل؟", "ولا عايز تقفل الحساب نهائيًا؟"]


@pytest.fixture
def force_escalation(monkeypatch):
    """نثبّت أن الرسالة ستُصعَّد (ثقة أقل من العتبة) لنختبر مسار التوضيح وحده."""
    monkeypatch.setattr(service, "ESCALATE_BELOW", 1.01)


def _fake_clarify(result):
    return lambda question, chunks: result


# ---------- قراءة رد الـ LLM ----------

def test_parse_caps_at_three_and_drops_sensitive_questions():
    content = ('{"out_of_scope": false, "questions": ["س1 عن التحويل؟", "ابعتلنا كود التحقق؟", '
               '"إيه الباسورد بتاعك؟", "س2 عن الوقت؟", "س3 عن المبلغ؟", "س4 زيادة؟"]}')
    parsed = common.parse_clarification(content)
    assert parsed == {"out_of_scope": False, "questions": ["س1 عن التحويل؟", "س2 عن الوقت؟", "س3 عن المبلغ؟"]}


def test_parse_out_of_scope_and_plain_text_and_garbage():
    assert common.parse_clarification('{"out_of_scope": true, "questions": []}') == {"out_of_scope": True, "questions": []}
    assert common.parse_clarification("1. إمتى حولت الفلوس؟\n2) كان كام؟\nشكرًا")["questions"] == ["إمتى حولت الفلوس؟", "كان كام؟"]
    assert common.parse_clarification("") is None
    assert common.parse_clarification("مش فاهم") is None
    assert common.parse_clarification('{"out_of_scope": false, "questions": ["رقم البطاقة كام؟"]}') is None


# ---------- قرار الخدمة ----------

def test_vague_message_gets_clarifying_questions(client, force_escalation, monkeypatch):
    monkeypatch.setattr(common, "generate_clarifying_questions", _fake_clarify({"out_of_scope": False, "questions": QUESTIONS}))
    body = client.post("/draft", json={"customer_message": VAGUE}, headers=bearer("sara.ahmed")).json()
    assert body["action"] == "clarify"
    assert body["clarifying_questions"] == QUESTIONS
    assert all(q in body["draft"] for q in QUESTIONS) and body["reason"]
    assert body["citations"]


def test_out_of_scope_is_escalated_without_questions(client, force_escalation, monkeypatch):
    monkeypatch.setattr(common, "generate_clarifying_questions", _fake_clarify({"out_of_scope": True, "questions": []}))
    body = client.post("/draft", json={"customer_message": "عندكم عروض على الموبايلات؟"}, headers=bearer("sara.ahmed")).json()
    assert body["action"] == "escalate" and body["clarifying_questions"] == [] and body["draft"] == ""


def test_llm_unavailable_falls_back_to_escalation(client, force_escalation, monkeypatch):
    monkeypatch.setattr(common, "generate_clarifying_questions", _fake_clarify(None))
    body = client.post("/draft", json={"customer_message": VAGUE}, headers=bearer("sara.ahmed")).json()
    assert body["action"] == "escalate" and body["reason"]


# ---------- رحلة كاملة: أسئلة ← رد العميل ← مسودة من المحادثة ----------

def test_full_clarification_round_trip(client, force_escalation, monkeypatch):
    monkeypatch.setattr(common, "generate_clarifying_questions", _fake_clarify({"out_of_scope": False, "questions": QUESTIONS}))
    sent = {}
    monkeypatch.setattr(service, "send_reply_email", lambda to_email, subject, body_text: (sent.update(body=body_text), (True, None))[1])

    ticket = client.post("/submit-ticket", json={"customer_message": VAGUE, "customer_email": "a@b.co"}).json()
    draft = client.post("/draft", json={"ticket_id": ticket["ticket_id"]}, headers=bearer("sara.ahmed")).json()
    assert draft["action"] == "clarify"

    res = client.post(f"/tickets/{ticket['ticket_id']}/resolve", headers=bearer("sara.ahmed"),
                      json={"final_text": draft["draft"], "resolution": "clarify"}).json()
    assert res["email_sent"] and res["ticket"]["status"] == "awaiting_customer"
    assert res["reply_link"] in sent["body"]                       # الرابط داخل الإيميل
    token = res["reply_link"].split("t=")[1]

    # التذكرة تختفي من قائمة الموظف أثناء انتظار العميل
    pending = client.get("/tickets", headers=bearer("sara.ahmed")).json()
    assert ticket["ticket_id"] not in [t["ticket_id"] for t in pending]

    view = client.get(f"/reply/{token}").json()
    assert view["can_reply"] and QUESTIONS[0] in view["questions"]
    assert "a@b.co" not in str(view)                              # لا بيانات غير الأسئلة

    assert client.post(f"/reply/{token}", json={"message": "الحساب اتجمد بعد ما كتبت الباسورد غلط"}).status_code == 200
    assert client.post(f"/reply/{token}", json={"message": "تاني"}).status_code == 409   # رد واحد فقط

    pending = client.get("/tickets", headers=bearer("sara.ahmed")).json()
    back = next(t for t in pending if t["ticket_id"] == ticket["ticket_id"])
    assert [m["sender"] for m in back["messages"]] == ["customer", "agent", "customer"]

    # جولة توضيح واحدة فقط: المسودة التالية لا تسأل مرة ثانية، وتُبنى من المحادثة كلها
    seen = {}
    monkeypatch.setattr(service, "handle_request", lambda msg, allow_clarify=True: (seen.update(msg=msg, allow=allow_clarify), service.DraftResponse(
        draft="x", citations=[], confidence=0.9, action="needs_review", cached=False, latency_ms=0))[1])
    client.post("/draft", json={"ticket_id": ticket["ticket_id"]}, headers=bearer("sara.ahmed"))
    assert seen["allow"] is False
    assert VAGUE in seen["msg"] and "توضيح العميل: الحساب اتجمد" in seen["msg"]


def test_failed_email_keeps_ticket_pending_and_returns_link(client):
    ticket = client.post("/submit-ticket", json={"customer_message": VAGUE, "customer_email": "a@b.co"}).json()
    res = client.post(f"/tickets/{ticket['ticket_id']}/resolve", headers=bearer("sara.ahmed"),
                      json={"final_text": "أسئلة", "resolution": "clarify"}).json()   # SendGrid غير مُهيَّأ في الاختبارات
    assert not res["email_sent"] and res["ticket"]["status"] == "pending"
    assert res["reply_link"] and "/app/reply.html?t=" in res["reply_link"]


def test_closed_ticket_cannot_be_resolved_again(client, monkeypatch):
    sent = []
    monkeypatch.setattr(service, "send_reply_email", lambda to_email, subject, body_text: (sent.append(body_text), (True, None))[1])
    ticket = client.post("/submit-ticket", json={"customer_message": VAGUE, "customer_email": "a@b.co"}).json()
    url, headers = f"/tickets/{ticket['ticket_id']}/resolve", bearer("sara.ahmed")
    assert client.post(url, headers=headers, json={"final_text": "رد", "resolution": "send"}).json()["ticket"]["status"] == "sent"
    for resolution in ("send", "clarify", "escalate"):
        assert client.post(url, headers=headers, json={"final_text": "رد", "resolution": resolution}).status_code == 409
    assert len(sent) == 1                                          # لا إيميل مكرر للعميل


def _ids_in(path) -> set[str]:
    import sqlite3
    if not path.exists():
        return set()
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT ticket_id FROM tickets")}


def test_only_follow_up_tickets_are_stored_durably(client, monkeypatch):
    """جديدة ← محلي فقط | أسئلة توضيحية ← تُنقل للمتابعة | إغلاق ← تُحذف من المتابعة."""
    monkeypatch.setattr(service, "send_reply_email", lambda to_email, subject, body_text: (True, None))
    headers = bearer("sara.ahmed")
    ticket = client.post("/submit-ticket", json={"customer_message": VAGUE, "customer_email": "a@b.co"}).json()
    tid = ticket["ticket_id"]
    assert tid in _ids_in(ticket_store.DB_PATH) and tid not in _ids_in(ticket_store.FOLLOWUP_DB_PATH)

    res = client.post(f"/tickets/{tid}/resolve", headers=headers, json={"final_text": "سؤال؟", "resolution": "clarify"}).json()
    assert res["ticket"]["status"] == "awaiting_customer"
    assert tid not in _ids_in(ticket_store.DB_PATH) and tid in _ids_in(ticket_store.FOLLOWUP_DB_PATH)

    # "إعادة تشغيل الحاوية": المخزن المحلي يُمسح، والمتابعة ورابط الرد يبقيان
    ticket_store.DB_PATH.unlink()
    token = res["reply_link"].split("t=")[1]
    assert client.post(f"/reply/{token}", json={"message": "توضيح"}).status_code == 200
    back = ticket_store.get_ticket(tid)
    assert back["status"] == "pending" and [m["sender"] for m in back["messages"]] == ["customer", "agent", "customer"]
    assert tid in [t["ticket_id"] for t in client.get("/tickets", headers=headers).json()]

    res = client.post(f"/tickets/{tid}/resolve", headers=headers, json={"final_text": "الرد النهائي", "resolution": "send"}).json()
    assert res["ticket"]["status"] == "sent" and res["ticket"]["messages"][-1]["text"] == "الرد النهائي"
    assert tid not in _ids_in(ticket_store.FOLLOWUP_DB_PATH)      # لا أرشيف
    assert ticket_store.get_ticket(tid) is None


def test_d1_store_speaks_the_worker_protocol(monkeypatch, tmp_path):
    """مخزن D1 يرسل الجمل بصيغة الـ Worker (cloudflare/src/index.js)؛ هنا "Worker"
    وهمي ينفّذها على SQLite ويرد بنفس شكل D1: [{rows, changes}]."""
    backing = ticket_store._SQLiteStore(tmp_path / "fake_d1.db")
    seen = []

    class FakeResponse:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data

    def fake_post(url, json, timeout):
        seen.append(url)
        return FakeResponse(backing.run([(s["sql"], s["params"]) for s in json["statements"]]))

    monkeypatch.setattr(ticket_store.httpx, "post", fake_post)
    monkeypatch.setattr(ticket_store, "D1_URL", "http://d1.sanad/query")

    t = ticket_store.create_ticket("رسالة", "a@b.co")
    ticket_store.start_follow_up(t["ticket_id"], "سؤال؟")
    assert seen and set(seen) == {"http://d1.sanad/query"}
    assert ticket_store.get_ticket(t["ticket_id"])["status"] == "awaiting_customer"
    assert ticket_store.set_status(t["ticket_id"], "pending", expected={"awaiting_customer"})
    assert not ticket_store.set_status(t["ticket_id"], "pending", expected={"awaiting_customer"})
    assert ticket_store.close_ticket(t["ticket_id"], "escalated")["status"] == "escalated"
    assert ticket_store.get_ticket(t["ticket_id"]) is None


def test_set_status_expected_is_conditional():
    t = ticket_store.create_ticket("رسالة", None)
    assert not ticket_store.set_status(t["ticket_id"], "pending", expected={"awaiting_customer"})
    assert ticket_store.set_status(t["ticket_id"], "awaiting_customer", expected={"pending"})
    assert ticket_store.set_status(t["ticket_id"], "pending", expected={"awaiting_customer"})
    assert not ticket_store.set_status(t["ticket_id"], "pending", expected={"awaiting_customer"})   # الثاني يفشل


def test_reply_token_security(client):
    assert client.get("/reply/not-a-real-token").status_code == 404
    assert client.post("/reply/not-a-real-token", json={"message": "x"}).status_code == 404
    ticket = client.post("/submit-ticket", json={"customer_message": VAGUE}).json()
    token = ticket_store.get_ticket(ticket["ticket_id"])["reply_token"]
    assert len(token) >= 30                                        # غير قابل للتخمين
    # تذكرة لم تُرسل لها أسئلة: لا يمكن الرد عليها
    assert client.get(f"/reply/{token}").json()["can_reply"] is False
    assert client.post(f"/reply/{token}", json={"message": "x"}).status_code == 409


def test_tickets_survive_restart():
    """التخزين في SQLite: التذكرة موجودة بعد "إعادة تشغيل" (اتصال جديد)."""
    t = ticket_store.create_ticket("رسالة", "a@b.co")
    import sqlite3
    with sqlite3.connect(ticket_store.DB_PATH) as conn:          # قراءة مستقلة من الملف نفسه
        assert conn.execute("SELECT COUNT(*) FROM tickets WHERE ticket_id = ?", (t["ticket_id"],)).fetchone()[0] == 1
    assert ticket_store.get_ticket(t["ticket_id"])["customer_message"] == "رسالة"
    assert ticket_store.DB_PATH.exists()
