// أدوات الاختبار: D1 وهمي فوق node:sqlite، وWorkers AI وهمي (embedding بتجزئة الكلمات)، وASSETS وهمي.
import { DatabaseSync } from "node:sqlite";

export function fakeD1() {
	const db = new DatabaseSync(":memory:");
	const wrap = (sql, params = []) => ({
		bind: (...p) => wrap(sql, p),
		async all() {
			const rows = db.prepare(sql).all(...params);
			return { results: rows, meta: {} };
		},
		async first() {
			return db.prepare(sql).get(...params) ?? null;
		},
		async run() {
			const r = db.prepare(sql).run(...params);
			return { meta: { changes: Number(r.changes) } };
		},
		_exec() {
			const isSelect = /^\s*select/i.test(sql);
			if (isSelect) return { results: db.prepare(sql).all(...params), meta: {} };
			const r = db.prepare(sql).run(...params);
			return { results: [], meta: { changes: Number(r.changes) } };
		},
	});
	return {
		prepare: (sql) => wrap(sql),
		async batch(stmts) {
			db.exec("BEGIN");
			try {
				const out = stmts.map((s) => s._exec());
				db.exec("COMMIT");
				return out;
			} catch (e) {
				db.exec("ROLLBACK");
				throw e;
			}
		},
	};
}

// embedding وهمي: تجزئة الكلمات/الأحرف إلى 256 بُعدًا (تشابه ≈ تداخل الكلمات).
export function fakeAI() {
	const calls = [];
	return {
		calls,
		async run(model, { text }) {
			calls.push(text.length);
			return {
				shape: [text.length, 256],
				data: text.map((t) => {
					const v = new Array(256).fill(0);
					for (const w of t.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean)) {
						let h = 2166136261;
						for (const ch of w) h = Math.imul(h ^ ch.codePointAt(0), 16777619) >>> 0;
						v[h % 256] += 1;
					}
					return v;
				}),
			};
		},
	};
}

export function makeEnv(extra = {}) {
	return {
		DB: fakeD1(),
		AI: fakeAI(),
		ASSETS: { fetch: async (req) => new Response(`asset:${new URL(req.url).pathname}`, { status: new URL(req.url).pathname === "/missing.html" ? 404 : 200 }) },
		SANAD_JWT_SECRET: "x".repeat(48),
		...extra,
	};
}

export const req = (path, { method = "GET", body, headers = {} } = {}) =>
	new Request("https://sanad.test" + path, {
		method,
		headers: { "content-type": "application/json", ...headers },
		body: body === undefined ? undefined : JSON.stringify(body),
	});
