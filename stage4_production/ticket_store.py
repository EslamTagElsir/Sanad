"""
stage4_production/ticket_store.py — تخزين التذاكر والمحادثات
-------------------------------------------------------------
مخزنان بنفس الجداول (SQL واحد):

  - المخزن المحلي: ملف SQLite داخل الحاوية (data/tickets.db). كل تذكرة جديدة تبدأ
    هنا. مؤقت عمدًا: على Cloudflare يُمسح مع نوم الحاوية أو إعادة تشغيلها، فتضيع
    التذاكر التي لم يفتحها موظف بعد (قرار مقبول لتقليل التخزين).
  - مخزن المتابعة: التذاكر التي أُرسلت فيها أسئلة توضيحية للعميل فقط، لأن العميل
    قد يرد من رابط الإيميل بعد ساعات ويجب أن تبقى. على Cloudflare هو قاعدة D1
    (SANAD_D1_URL، عبر الـ Worker — راجع cloudflare/src/index.js)، ومحليًا ملف
    SQLite منفصل (data/followups.db).

دورة الحياة: جديدة (محلي) ← إرسال أسئلة توضيحية: تُنقل للمتابعة (start_follow_up)
← رد العميل ← رد نهائي أو تصعيد: تُحذف من المتابعة (close_ticket). لا أرشيف.

reply_token رمز عشوائي طويل يُرسل في رابط الإيميل ليرد العميل على نفس التذكرة دون
تسجيل دخول، ولا يمكن تخمينه.
"""

import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx

_DATA = Path(__file__).parent.parent / "data"
DB_PATH = Path(os.environ.get("SANAD_DB_PATH") or _DATA / "tickets.db")
FOLLOWUP_DB_PATH = Path(os.environ.get("SANAD_FOLLOWUP_DB_PATH") or _DATA / "followups.db")
# على Cloudflare: http://d1.sanad/query (يعترضه الـ Worker وينفّذه على D1).
D1_URL = os.environ.get("SANAD_D1_URL", "").strip()
D1_TIMEOUT_S = 10

# pending: بانتظار الموظف | awaiting_customer: أُرسلت أسئلة توضيحية وبانتظار رد العميل
# sent: أُرسل الرد النهائي | escalated: صُعّدت لفريق أعلى
STATUSES = {"pending", "awaiting_customer", "sent", "escalated"}

# نفس المخطط في cloudflare/migrations/0001_init.sql (D1 لا يُنشئه وقت التشغيل).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      TEXT PRIMARY KEY,
    customer_email TEXT,
    status         TEXT NOT NULL,
    reply_token    TEXT NOT NULL UNIQUE,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL REFERENCES tickets(ticket_id),
    sender     TEXT NOT NULL CHECK (sender IN ('customer', 'agent')),
    kind       TEXT NOT NULL CHECK (kind IN ('message', 'clarification')),
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id, id);
"""


# ---------------------------------------------------------------------------
# المخزنان: كلاهما ينفّذ قائمة جمل SQL كمعاملة واحدة ويرجع لكل جملة
# {"rows": [...], "changes": عدد الصفوف المتغيرة}.
# ---------------------------------------------------------------------------
class _SQLiteStore:
    def __init__(self, path: Path):
        self.path = path

    def run(self, statements: list[tuple[str, list]]) -> list[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_SCHEMA)
            out = []
            for sql, params in statements:
                cur = conn.execute(sql, params)
                out.append({"rows": [dict(r) for r in cur.fetchall()], "changes": cur.rowcount})
            conn.commit()
            return out
        finally:
            conn.close()


class _D1Store:
    """D1 عبر الـ Worker: الجمل تُرسل دفعة واحدة وD1 ينفّذها كمعاملة (batch)."""

    def __init__(self, url: str):
        self.url = url

    def run(self, statements: list[tuple[str, list]]) -> list[dict]:
        res = httpx.post(self.url, timeout=D1_TIMEOUT_S,
                         json={"statements": [{"sql": s, "params": p} for s, p in statements]})
        res.raise_for_status()
        return res.json()


def _local() -> _SQLiteStore:
    return _SQLiteStore(DB_PATH)


def _followup():
    return _D1Store(D1_URL) if D1_URL else _SQLiteStore(FOLLOWUP_DB_PATH)


def _stores():
    return [_local(), _followup()]


def _is_local(store) -> bool:
    return isinstance(store, _SQLiteStore) and store.path == DB_PATH


# ---------------------------------------------------------------------------
# قراءة
# ---------------------------------------------------------------------------
_MSG_COLS = "ticket_id, sender, kind, text, created_at"


def _assemble(ticket_rows: list[dict], msg_rows: list[dict]) -> list[dict]:
    by_ticket: dict[str, list] = {}
    for m in msg_rows:
        by_ticket.setdefault(m["ticket_id"], []).append({k: m[k] for k in ("sender", "kind", "text", "created_at")})
    tickets = []
    for row in ticket_rows:
        t = dict(row)
        t["messages"] = by_ticket.get(t["ticket_id"], [])
        t["customer_message"] = next((m["text"] for m in t["messages"] if m["sender"] == "customer"), "")
        tickets.append(t)
    return tickets


def _query(store, where: str, params: list) -> list[dict]:
    """التذاكر المطابقة ورسائلها في طلب واحد (مهم لـ D1: رحلة شبكة واحدة)."""
    tickets, msgs = store.run([
        (f"SELECT * FROM tickets WHERE {where} ORDER BY updated_at", params),
        (f"SELECT {_MSG_COLS} FROM messages WHERE ticket_id IN (SELECT ticket_id FROM tickets WHERE {where}) ORDER BY id", params),
    ])
    return _assemble(tickets["rows"], msgs["rows"])


def _find(where: str, params: list) -> tuple[Optional[dict], Optional[object]]:
    """(التذكرة، مخزنها) — المحلي أولًا (أرخص)، ثم المتابعة."""
    for store in _stores():
        found = _query(store, where, params)
        if found:
            return found[0], store
    return None, None


def get_ticket(ticket_id: str) -> Optional[dict]:
    return _find("ticket_id = ?", [ticket_id])[0]


def get_by_reply_token(token: str) -> Optional[dict]:
    return _find("reply_token = ?", [token])[0]


def list_tickets(status: str = "pending") -> list[dict]:
    where, params = ("1 = 1", []) if status == "all" else ("status = ?", [status])
    merged = [t for store in _stores() for t in _query(store, where, params)]
    return sorted(merged, key=lambda t: t["updated_at"])


# ---------------------------------------------------------------------------
# كتابة
# ---------------------------------------------------------------------------
def _insert_message(ticket_id: str, sender: str, kind: str, text: str, now: float) -> tuple[str, list]:
    return ("INSERT INTO messages (ticket_id, sender, kind, text, created_at) VALUES (?, ?, ?, ?, ?)",
            [ticket_id, sender, kind, text, now])


def create_ticket(customer_message: str, customer_email: Optional[str]) -> dict:
    now = time.time()
    ticket_id = secrets.token_hex(6)
    _local().run([
        ("INSERT INTO tickets (ticket_id, customer_email, status, reply_token, created_at, updated_at) "
         "VALUES (?, ?, 'pending', ?, ?, ?)", [ticket_id, customer_email, secrets.token_urlsafe(24), now, now]),
        _insert_message(ticket_id, "customer", "message", customer_message, now),
    ])
    return get_ticket(ticket_id)


def add_message(ticket_id: str, sender: str, text: str, kind: str = "message") -> None:
    _, store = _find("ticket_id = ?", [ticket_id])
    if store is None:
        return
    now = time.time()
    store.run([
        _insert_message(ticket_id, sender, kind, text, now),
        ("UPDATE tickets SET updated_at = ? WHERE ticket_id = ?", [now, ticket_id]),
    ])


def set_status(ticket_id: str, status: str, expected: Optional[set[str]] = None) -> bool:
    """expected: الحالات المسموح الانتقال منها. الفحص والتغيير في جملة UPDATE واحدة
    (ذرّي)، فطلبان متزامنان لا يمرّان معًا. يُرجع False إن لم تتغير الحالة."""
    assert status in STATUSES, status
    _, store = _find("ticket_id = ?", [ticket_id])
    if store is None:
        return False
    sql, params = "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?", [status, time.time(), ticket_id]
    if expected is not None:
        sql += f" AND status IN ({', '.join('?' * len(expected))})"
        params += sorted(expected)
    return store.run([(sql, params)])[0]["changes"] > 0


def start_follow_up(ticket_id: str, questions_text: str) -> Optional[dict]:
    """أُرسلت أسئلة توضيحية: التذكرة تصبح awaiting_customer وتُنقل (بمحادثتها كلها)
    لمخزن المتابعة الدائم. الكتابة في المتابعة أولًا ثم الحذف من المحلي، فلا تضيع
    التذكرة لو فشل أحدهما."""
    ticket, store = _find("ticket_id = ?", [ticket_id])
    if ticket is None:
        return None
    now = time.time()
    followup = _followup()
    if _is_local(store):
        followup.run(
            [("INSERT INTO tickets (ticket_id, customer_email, status, reply_token, created_at, updated_at) "
              "VALUES (?, ?, 'awaiting_customer', ?, ?, ?)",
              [ticket_id, ticket["customer_email"], ticket["reply_token"], ticket["created_at"], now])]
            + [_insert_message(ticket_id, m["sender"], m["kind"], m["text"], m["created_at"]) for m in ticket["messages"]]
            + [_insert_message(ticket_id, "agent", "clarification", questions_text, now)]
        )
        store.run([("DELETE FROM messages WHERE ticket_id = ?", [ticket_id]),
                   ("DELETE FROM tickets WHERE ticket_id = ?", [ticket_id])])
    else:   # موجودة في المتابعة أصلًا (جولة أسئلة ثانية أرسلها الموظف يدويًا)
        followup.run([
            _insert_message(ticket_id, "agent", "clarification", questions_text, now),
            ("UPDATE tickets SET status = 'awaiting_customer', updated_at = ? WHERE ticket_id = ?", [now, ticket_id]),
        ])
    return get_ticket(ticket_id)


def close_ticket(ticket_id: str, status: str, final_text: Optional[str] = None) -> Optional[dict]:
    """رد نهائي (sent) أو تصعيد (escalated). يُرجع التذكرة بحالتها النهائية. تذكرة
    المتابعة تُحذف من مخزنها الدائم (لا أرشيف)؛ المحلية تبقى حتى يُمسح المخزن المحلي."""
    assert status in ("sent", "escalated"), status
    ticket, store = _find("ticket_id = ?", [ticket_id])
    if ticket is None:
        return None
    now = time.time()
    if final_text is not None:
        ticket["messages"].append({"sender": "agent", "kind": "message", "text": final_text, "created_at": now})
    ticket.update(status=status, updated_at=now)
    if _is_local(store):
        store.run(([_insert_message(ticket_id, "agent", "message", final_text, now)] if final_text is not None else [])
                  + [("UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?", [status, now, ticket_id])])
    else:
        store.run([("DELETE FROM messages WHERE ticket_id = ?", [ticket_id]),
                   ("DELETE FROM tickets WHERE ticket_id = ?", [ticket_id])])
    return ticket


def conversation_for_retrieval(ticket: dict) -> str:
    """نص البحث والمسودة: رسالة العميل الأولى + توضيحاته اللاحقة (بدون كلام الموظف،
    حتى لا تنحرف المصادر المسترجعة نحو صياغة أسئلتنا نحن)."""
    customer = [m["text"] for m in ticket["messages"] if m["sender"] == "customer"]
    if len(customer) <= 1:
        return customer[0] if customer else ""
    return customer[0] + "\n" + "\n".join(f"توضيح العميل: {t}" for t in customer[1:])


def had_clarification(ticket: dict) -> bool:
    """جولة توضيح واحدة فقط لكل تذكرة: إن سألنا من قبل ولم يتضح الأمر، يُصعَّد."""
    return any(m["kind"] == "clarification" for m in ticket["messages"])
