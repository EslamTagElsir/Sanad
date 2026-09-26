-- Follow-up tickets only (clarifying questions sent, waiting for / answered by
-- the customer). Same schema as stage4_production/ticket_store.py _SCHEMA.
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
