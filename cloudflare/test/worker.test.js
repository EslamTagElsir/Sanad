import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "../src/index.js";
import { hashPassword, verifyPassword } from "../src/lib/auth.js";
import { fitCalibration, predictConfidence } from "../src/lib/retrieval.js";
import { parseClarification } from "../src/lib/llm.js";
import { decideAction } from "../src/lib/service.js";
import { makeEnv, req } from "./helpers.js";

const call = async (env, path, opts) => {
	const res = await worker.fetch(req(path, opts), env);
	const text = await res.text();
	let data;
	try { data = JSON.parse(text); } catch { data = text; }
	return { status: res.status, data };
};

async function envWithUser() {
	const password_hash = await hashPassword("correct-horse-battery", 1000);
	return makeEnv({ SANAD_USERS_JSON: JSON.stringify([{ username: "sara", display_name: "سارة", role: "employee", password_hash }]) });
}

async function login(env, password = "correct-horse-battery") {
	return call(env, "/login", { method: "POST", body: { username: "sara", password } });
}

test("password hashing roundtrip", async () => {
	const h = await hashPassword("secret-pass-123", 1000);
	assert.equal(await verifyPassword("secret-pass-123", h), true);
	assert.equal(await verifyPassword("wrong", h), false);
	assert.equal(await verifyPassword("x", "bcrypt$nope"), false);
});

test("login, JWT and role gating", async () => {
	const env = await envWithUser();
	assert.equal((await login(env, "bad")).status, 401);
	const ok = await login(env);
	assert.equal(ok.status, 200);
	assert.equal(ok.data.role, "employee");
	const auth = { authorization: "Bearer " + ok.data.access_token };
	assert.equal((await call(env, "/tickets", { headers: auth })).status, 200);
	assert.equal((await call(env, "/tickets")).status, 401);
	assert.equal((await call(env, "/tickets", { headers: { authorization: "Bearer " + ok.data.access_token + "x" } })).status, 401);
	// مستخدم محذوف => توكنه يسقط فورًا
	const env2 = makeEnv({ ...env, SANAD_USERS_JSON: "[]", DB: env.DB });
	assert.equal((await call(env2, "/tickets", { headers: auth })).status, 401);
});

test("lockout after 5 failed logins", async () => {
	const env = await envWithUser();
	for (let i = 0; i < 5; i++) assert.equal((await login(env, "bad")).status, 401);
	assert.equal((await login(env)).status, 429);
});

test("missing secret => 503, api key and customer role", async () => {
	const env = makeEnv({ SANAD_EMPLOYEE_KEYS: "k1,k2", SANAD_CUSTOMER_KEYS: "c1" });
	assert.equal((await call(env, "/tickets", { headers: { "x-api-key": "k2" } })).status, 200);
	assert.equal((await call(env, "/tickets", { headers: { "x-api-key": "c1" } })).status, 403);
	assert.equal((await call(env, "/tickets", { headers: { "x-api-key": "nope" } })).status, 401);
	const env2 = makeEnv({ SANAD_JWT_SECRET: undefined });
	assert.equal((await call(env2, "/tickets", { headers: { authorization: "Bearer a.b.c" } })).status, 503);
});

test("ticket lifecycle: submit -> clarify -> customer reply -> send", async () => {
	const env = await envWithUser();
	env.SENDGRID_API_KEY = "sg"; env.SENDGRID_FROM_EMAIL = "support@example.com";
	const sent = [];
	const realFetch = globalThis.fetch;
	globalThis.fetch = async (url, init) => { sent.push({ url: String(url), body: JSON.parse(init.body) }); return new Response("", { status: 202 }); };
	try {
		const auth = { authorization: "Bearer " + (await login(env)).data.access_token };
		assert.equal((await call(env, "/submit-ticket", { method: "POST", body: { customer_message: "hi", customer_email: "bad" } })).status, 422);
		const t = (await call(env, "/submit-ticket", { method: "POST", body: { customer_message: "حولت فلوس ومش واصلة", customer_email: "a@b.co" } })).data;
		assert.equal(t.status, "pending");
		assert.equal((await call(env, "/tickets", { headers: auth })).data.length, 1);

		const r1 = await call(env, `/tickets/${t.ticket_id}/resolve`, { method: "POST", headers: auth, body: { final_text: "سؤال؟", resolution: "clarify" } });
		assert.equal(r1.status, 200);
		assert.equal(r1.data.email_sent, true);
		assert.equal(r1.data.ticket.status, "awaiting_customer");
		assert.match(r1.data.reply_link, /^https:\/\/sanad\.test\/app\/reply\.html\?t=/);
		assert.ok(sent[0].body.content[0].value.includes("للرد على الأسئلة"));

		const token = r1.data.reply_link.split("t=")[1];
		const view = await call(env, `/reply/${token}`);
		assert.equal(view.data.can_reply, true);
		assert.equal(view.data.questions, "سؤال؟");
		assert.equal((await call(env, `/reply/${token}`, { method: "POST", body: { message: "ايوه" } })).status, 200);
		assert.equal((await call(env, `/reply/${token}`, { method: "POST", body: { message: "تاني" } })).status, 409);
		assert.equal((await call(env, `/reply/${"z".repeat(70)}`)).status, 404);

		const pending = (await call(env, "/tickets", { headers: auth })).data;
		assert.equal(pending[0].messages.length, 3);
		const r2 = await call(env, `/tickets/${t.ticket_id}/resolve`, { method: "POST", headers: auth, body: { final_text: "تم", resolution: "send" } });
		assert.equal(r2.data.ticket.status, "sent");
		assert.equal((await call(env, `/tickets/${t.ticket_id}/resolve`, { method: "POST", headers: auth, body: { final_text: "x", resolution: "send" } })).status, 409);
	} finally { globalThis.fetch = realFetch; }
});

test("email failure keeps ticket pending", async () => {
	const env = await envWithUser();
	const auth = { authorization: "Bearer " + (await login(env)).data.access_token };
	const t = (await call(env, "/submit-ticket", { method: "POST", body: { customer_message: "x", customer_email: "a@b.co" } })).data;
	const r = await call(env, `/tickets/${t.ticket_id}/resolve`, { method: "POST", headers: auth, body: { final_text: "x", resolution: "clarify" } });
	assert.equal(r.data.email_sent, false);
	assert.match(r.data.email_error, /SENDGRID/);
	assert.equal(r.data.ticket.status, "pending");
	assert.ok(r.data.reply_link);
});

test("draft: index init in small steps, then real draft (no LLM key)", async () => {
	const env = await envWithUser();
	const auth = { authorization: "Bearer " + (await login(env)).data.access_token };
	const t = (await call(env, "/submit-ticket", { method: "POST", body: { customer_message: "How long does identity verification take after I upload my ID?" } })).data;
	let r = await call(env, "/draft", { method: "POST", headers: auth, body: { ticket_id: t.ticket_id } });
	assert.equal(r.status, 503);
	assert.equal(r.data.init_pending, true);
	let steps = 0, p;
	do { p = (await call(env, "/admin/init", { method: "POST", headers: auth })).data; steps++; } while (!p.ready && steps < 40);
	assert.equal(p.ready, true, JSON.stringify(p));
	assert.ok(steps < 30, "steps=" + steps);
	assert.equal((await call(env, "/health")).data.ready, true);
	r = await call(env, "/draft", { method: "POST", headers: auth, body: { ticket_id: t.ticket_id } });
	assert.equal(r.status, 200, JSON.stringify(r.data));
	assert.ok(["send_ready", "needs_review", "escalate", "clarify"].includes(r.data.action));
	assert.ok(r.data.confidence >= 0 && r.data.confidence <= 1);
	assert.ok(Array.isArray(r.data.citations));
	// جاهز بعد isolate جديد (الفهرس من D1)
	const again = await call(env, "/draft", { method: "POST", headers: auth, body: { customer_message: "ما المستندات المطلوبة لتوثيق الحساب؟" } });
	assert.equal(again.status, 200);
	assert.equal((await call(env, "/draft", { method: "POST", headers: auth, body: {} })).status, 422);
});

test("calibration fit is sensible and matches sklearn-style output", () => {
	const rows = [];
	for (let i = 0; i < 40; i++) rows.push({ dense: 0.2 + i * 0.015, label: i > 20 ? 1 : 0 });
	const cal = fitCalibration(rows);
	assert.equal(cal.mode, "lr");
	const lo = predictConfidence(cal, { chunk_id: "a" }, { a: { dense: 0.25 } });
	const hi = predictConfidence(cal, { chunk_id: "a" }, { a: { dense: 0.75 } });
	assert.ok(lo < 0.2 && hi > 0.8, `${lo} ${hi}`);
	assert.equal(fitCalibration([{ dense: 1, label: 1 }, { dense: 0.5, label: 1 }]).mode, "raw");
	assert.equal(predictConfidence({ mode: "raw" }, { chunk_id: "a" }, { a: { dense: 1.7 } }), 1);
	assert.equal(predictConfidence(cal, null, {}), 0);
});

test("decide action thresholds", () => {
	assert.equal(decideAction(0.9, { source_type: "kb" }), "send_ready");
	assert.equal(decideAction(0.9, { source_type: "past_ticket" }), "needs_review");
	assert.equal(decideAction(0.5, { source_type: "kb" }), "needs_review");
	assert.equal(decideAction(0.3, { source_type: "kb" }), "escalate");
	assert.equal(decideAction(0.9, undefined), "escalate");
});

test("clarification parsing drops sensitive + caps at 3", () => {
	const r = parseClarification('{"out_of_scope": false, "questions": ["1. ما الخدمة؟", "ما كود التحقق؟", "ما المبلغ؟", "متى؟", "أين؟"]}');
	assert.deepEqual(r.questions, ["ما الخدمة؟", "ما المبلغ؟", "متى؟"]);
	assert.deepEqual(parseClarification('{"out_of_scope": true, "questions": []}'), { out_of_scope: true, questions: [] });
	assert.equal(parseClarification(""), null);
	assert.deepEqual(parseClarification("ما المشكلة؟\nشكرا").questions, ["ما المشكلة؟"]);
});

test("static assets + routing", async () => {
	const env = makeEnv();
	const a = await worker.fetch(req("/app/agent.html"), env);
	assert.equal(await a.text(), "asset:/agent.html");
	assert.equal(await (await worker.fetch(req("/app/"), env)).text(), "asset:/index.html");
	assert.equal((await worker.fetch(req("/app/missing.html"), env)).status, 404);
	assert.equal((await worker.fetch(req("/"), env)).status, 302);
	assert.equal((await call(env, "/nope")).status, 404);
	assert.equal((await call(env, "/login", { method: "POST", body: { username: 1 } })).status, 422);
});
