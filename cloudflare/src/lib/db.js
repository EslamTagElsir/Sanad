// مخطط D1: يُنشأ تلقائيًا عند أول استخدام (النشر عبر Workers Builds لا يشغّل migrations).
// نفس المخطط في migrations/ للمن يفضّل wrangler d1 migrations apply.
const SCHEMA = [
	`CREATE TABLE IF NOT EXISTS tickets (
		ticket_id      TEXT PRIMARY KEY,
		customer_email TEXT,
		status         TEXT NOT NULL,
		reply_token    TEXT NOT NULL UNIQUE,
		created_at     REAL NOT NULL,
		updated_at     REAL NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS messages (
		id         INTEGER PRIMARY KEY AUTOINCREMENT,
		ticket_id  TEXT NOT NULL REFERENCES tickets(ticket_id),
		sender     TEXT NOT NULL CHECK (sender IN ('customer', 'agent')),
		kind       TEXT NOT NULL CHECK (kind IN ('message', 'clarification')),
		text       TEXT NOT NULL,
		created_at REAL NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id, id)`,
	`CREATE TABLE IF NOT EXISTS login_failures (username TEXT NOT NULL, ts REAL NOT NULL)`,
	`CREATE INDEX IF NOT EXISTS idx_login_failures ON login_failures(username, ts)`,
	`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)`,
	`CREATE TABLE IF NOT EXISTS embeddings (
		id    TEXT PRIMARY KEY,
		kind  TEXT NOT NULL,
		scale REAL NOT NULL,
		vec   TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS calib_rows (item_id TEXT PRIMARY KEY, dense REAL NOT NULL, label INTEGER NOT NULL)`,
];

const _ready = new WeakMap(); // env.DB -> Promise (نسخة واحدة لكل isolate/قاعدة)
export function ensureSchema(env) {
	// الفشل لا يُخزَّن (تُعاد المحاولة).
	if (!_ready.has(env.DB)) {
		_ready.set(
			env.DB,
			env.DB.batch(SCHEMA.map((sql) => env.DB.prepare(sql))).catch((err) => {
				_ready.delete(env.DB);
				throw err;
			}),
		);
	}
	return _ready.get(env.DB);
}
