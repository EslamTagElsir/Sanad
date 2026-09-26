// تخزين التذاكر والمحادثات في D1 (مخزن واحد دائم؛ على الخطة المجانية لا توجد حاوية بقرص مؤقت).
// reply_token رمز عشوائي طويل يُرسل في رابط الإيميل ليرد العميل على نفس التذكرة بلا تسجيل دخول.
import { nowSeconds, randomHex, randomUrlSafe } from "./util.js";

// pending: بانتظار الموظف | awaiting_customer: أُرسلت أسئلة توضيحية | sent: رد نهائي | escalated: صُعّدت
export const STATUSES = new Set(["pending", "awaiting_customer", "sent", "escalated"]);

function assemble(ticketRows, msgRows) {
	const byTicket = new Map();
	for (const m of msgRows) {
		if (!byTicket.has(m.ticket_id)) byTicket.set(m.ticket_id, []);
		byTicket.get(m.ticket_id).push({ sender: m.sender, kind: m.kind, text: m.text, created_at: m.created_at });
	}
	return ticketRows.map((t) => {
		const messages = byTicket.get(t.ticket_id) || [];
		return { ...t, messages, customer_message: messages.find((m) => m.sender === "customer")?.text ?? "" };
	});
}

// التذاكر المطابقة ورسائلها في رحلة واحدة (batch).
async function query(env, where, params) {
	const [tickets, msgs] = await env.DB.batch([
		env.DB.prepare(`SELECT * FROM tickets WHERE ${where} ORDER BY updated_at`).bind(...params),
		env.DB.prepare(
			`SELECT ticket_id, sender, kind, text, created_at FROM messages WHERE ticket_id IN (SELECT ticket_id FROM tickets WHERE ${where}) ORDER BY id`,
		).bind(...params),
	]);
	return assemble(tickets.results ?? [], msgs.results ?? []);
}

export async function getTicket(env, ticketId) {
	return (await query(env, "ticket_id = ?", [ticketId]))[0] ?? null;
}

export async function getByReplyToken(env, token) {
	return (await query(env, "reply_token = ?", [token]))[0] ?? null;
}

export async function listTickets(env, status = "pending") {
	return status === "all" ? query(env, "1 = 1", []) : query(env, "status = ?", [status]);
}

const insertMessage = (env, ticketId, sender, kind, text, at) =>
	env.DB.prepare("INSERT INTO messages (ticket_id, sender, kind, text, created_at) VALUES (?, ?, ?, ?, ?)").bind(
		ticketId,
		sender,
		kind,
		text,
		at,
	);

export async function createTicket(env, customerMessage, customerEmail) {
	const now = nowSeconds();
	const ticketId = randomHex(6);
	await env.DB.batch([
		env.DB.prepare(
			"INSERT INTO tickets (ticket_id, customer_email, status, reply_token, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?, ?)",
		).bind(ticketId, customerEmail ?? null, randomUrlSafe(24), now, now),
		insertMessage(env, ticketId, "customer", "message", customerMessage, now),
	]);
	return getTicket(env, ticketId);
}

export async function addMessage(env, ticketId, sender, text, kind = "message") {
	const now = nowSeconds();
	await env.DB.batch([
		insertMessage(env, ticketId, sender, kind, text, now),
		env.DB.prepare("UPDATE tickets SET updated_at = ? WHERE ticket_id = ?").bind(now, ticketId),
	]);
}

// expected: الحالات المسموح الانتقال منها. الفحص والتغيير في UPDATE واحدة (ذرّي)،
// فطلبان متزامنان لا يمرّان معًا. يُرجع false إن لم تتغير الحالة.
export async function setStatus(env, ticketId, status, expected = null) {
	if (!STATUSES.has(status)) throw new Error(`حالة غير معروفة: ${status}`);
	let sql = "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?";
	const params = [status, nowSeconds(), ticketId];
	if (expected) {
		const list = [...expected].sort();
		sql += ` AND status IN (${list.map(() => "?").join(", ")})`;
		params.push(...list);
	}
	const res = await env.DB.prepare(sql).bind(...params).run();
	return (res.meta?.changes ?? 0) > 0;
}

// أُرسلت أسئلة توضيحية: تسجَّل الرسالة وتصبح التذكرة awaiting_customer (معاملة واحدة).
export async function startFollowUp(env, ticketId, questionsText) {
	const now = nowSeconds();
	await env.DB.batch([
		insertMessage(env, ticketId, "agent", "clarification", questionsText, now),
		env.DB.prepare("UPDATE tickets SET status = 'awaiting_customer', updated_at = ? WHERE ticket_id = ?").bind(now, ticketId),
	]);
	return getTicket(env, ticketId);
}

// رد نهائي (sent) أو تصعيد (escalated).
export async function closeTicket(env, ticketId, status, finalText = null) {
	if (status !== "sent" && status !== "escalated") throw new Error(`حالة إغلاق غير صالحة: ${status}`);
	const now = nowSeconds();
	const stmts = [];
	if (finalText !== null) stmts.push(insertMessage(env, ticketId, "agent", "message", finalText, now));
	stmts.push(env.DB.prepare("UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?").bind(status, now, ticketId));
	await env.DB.batch(stmts);
	return getTicket(env, ticketId);
}

// نص البحث والمسودة: رسالة العميل الأولى + توضيحاته اللاحقة (بدون كلام الموظف).
export function conversationForRetrieval(ticket) {
	const customer = ticket.messages.filter((m) => m.sender === "customer").map((m) => m.text);
	if (customer.length <= 1) return customer[0] ?? "";
	return customer[0] + "\n" + customer.slice(1).map((t) => `توضيح العميل: ${t}`).join("\n");
}

// جولة توضيح واحدة فقط لكل تذكرة: إن سألنا من قبل ولم يتضح الأمر، يُصعَّد.
export const hadClarification = (ticket) => ticket.messages.some((m) => m.kind === "clarification");
