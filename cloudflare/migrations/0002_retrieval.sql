-- جداول الاسترجاع وقفل الدخول. الـ Worker ينشئها تلقائيًا (CREATE TABLE IF NOT EXISTS في
-- src/lib/db.js) فلا حاجة لتشغيل هذا الملف يدويًا؛ موجود للتوثيق ولمن يستخدم migrations.
CREATE TABLE IF NOT EXISTS login_failures (username TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_login_failures ON login_failures(username, ts);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, kind TEXT NOT NULL, scale REAL NOT NULL, vec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calib_rows (item_id TEXT PRIMARY KEY, dense REAL NOT NULL, label INTEGER NOT NULL);
