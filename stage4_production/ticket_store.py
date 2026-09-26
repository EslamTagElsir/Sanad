"""
stage4_production/ticket_store.py — تخزين التذاكر والمحادثات في SQLite
-----------------------------------------------------------------------
بديل الـ dict الذي كان في الذاكرة (كانت التذاكر تضيع مع كل إعادة تشغيل للسيرفر).
ملف واحد (data/tickets.db) بلا سيرفر قاعدة بيانات، ويتحمّل إعادة التشغيل — مهم
لروابط رد العميل على الأسئلة التوضيحية، إذ قد يرد بعد ساعات.

كل تذكرة لها محادثة (messages): أول رسالة من العميل، ثم ردود الموظف (رد نهائي أو
أسئلة توضيحية)، ثم توضيحات العميل. reply_token رمز عشوائي طويل يُرسل في رابط
الإيميل ليرد العميل على نفس التذكرة دون تسجيل دخول، ولا يمكن تخمينه.
"""

import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# SANAD_DB_PATH: على Hugging Face Spaces مع التخزين الدائم ضعه /data/tickets.db
# (بدونه القاعدة داخل الحاوية وتضيع مع كل إعادة تشغيل/نشر).
DB_PATH = Path(os.environ.get("SANAD_DB_PATH") or Path(__file__).parent.parent / "data" / "tickets.db")

# pending: بانتظار الموظف | awaiting_customer: أُرسلت أسئلة توضيحية وبانتظار رد العميل
# sent: أُرسل الرد النهائي | escalated: صُعّدت لفريق أعلى
STATUSES = {"pending", "awaiting_customer", "sent", "escalated"}

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


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _with_messages(conn, row) -> Optional[dict]:
    if row is None:
        return None
    ticket = dict(row)
    msgs = conn.execute(
        "SELECT sender, kind, text, created_at FROM messages WHERE ticket_id = ? ORDER BY id",
        (ticket["ticket_id"],),
    ).fetchall()
    ticket["messages"] = [dict(m) for m in msgs]
    ticket["customer_message"] = next((m["text"] for m in ticket["messages"] if m["sender"] == "customer"), "")
    return ticket


def create_ticket(customer_message: str, customer_email: Optional[str]) -> dict:
    now = time.time()
    ticket_id = secrets.token_hex(6)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_id, customer_email, status, reply_token, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?)",
            (ticket_id, customer_email, secrets.token_urlsafe(24), now, now),
        )
        conn.execute(
            "INSERT INTO messages (ticket_id, sender, kind, text, created_at) VALUES (?, 'customer', 'message', ?, ?)",
            (ticket_id, customer_message, now),
        )
        return _with_messages(conn, conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone())


def get_ticket(ticket_id: str) -> Optional[dict]:
    with _connect() as conn:
        return _with_messages(conn, conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone())


def get_by_reply_token(token: str) -> Optional[dict]:
    with _connect() as conn:
        return _with_messages(conn, conn.execute("SELECT * FROM tickets WHERE reply_token = ?", (token,)).fetchone())


def list_tickets(status: str = "pending") -> list[dict]:
    with _connect() as conn:
        if status == "all":
            rows = conn.execute("SELECT * FROM tickets ORDER BY updated_at").fetchall()
        else:
            rows = conn.execute("SELECT * FROM tickets WHERE status = ? ORDER BY updated_at", (status,)).fetchall()
        return [_with_messages(conn, r) for r in rows]


def add_message(ticket_id: str, sender: str, text: str, kind: str = "message") -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (ticket_id, sender, kind, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, sender, kind, text, now),
        )
        conn.execute("UPDATE tickets SET updated_at = ? WHERE ticket_id = ?", (now, ticket_id))


def set_status(ticket_id: str, status: str) -> None:
    assert status in STATUSES, status
    with _connect() as conn:
        conn.execute("UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?", (status, time.time(), ticket_id))


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
