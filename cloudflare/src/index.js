// Sanad على Cloudflare Workers (الخطة المجانية، بدون حاويات).
// نفس واجهة HTTP التي كانت في stage4_production/service.py (FastAPI) فتعمل الصفحات
// الثابتة (stage4_production/static) بلا تغيير، لكن الاسترجاع الآن عبر Workers AI
// والتخزين في D1. راجع README.md (قسم النشر على Cloudflare).
import {
	authenticate,
	clearFailedLogins,
	createAccessToken,
	isLockedOut,
	recordFailedLogin,
	requireEmployee,
} from "./lib/auth.js";
import { ensureSchema } from "./lib/db.js";
import { sendReplyEmail } from "./lib/email.js";
import { handleRequest } from "./lib/service.js";
import { initStep, isReady } from "./lib/retrieval.js";
import * as store from "./lib/tickets.js";
import { HttpError, json } from "./lib/util.js";

const MAX_MESSAGE_CHARS = 4000;
const EMAIL_RE = /^[^@\s<>"'`]+@[^@\s<>"'`]+\.[^@\s<>"'`]+$/;
const CLOSED_STATUSES = new Set(["sent", "escalated"]);

async function readJson(request) {
	try {
		const body = await request.json();
		if (body && typeof body === "object" && !Array.isArray(body)) return body;
	} catch {
		/* يسقط للخطأ أدناه */
	}
	throw new HttpError(422, "جسم الطلب يجب أن يكون JSON صالحًا");
}

function str(body, field, { min = 0, max = Infinity, required = true } = {}) {
	const v = body[field];
	if (v === undefined || v === null) {
		if (required) throw new HttpError(422, `الحقل ${field} مطلوب`);
		return null;
	}
	if (typeof v !== "string" || v.length < min || v.length > max) throw new HttpError(422, `الحقل ${field} غير صالح`);
	return v;
}

function publicUrl(env, request) {
	const configured = String(env.SANAD_PUBLIC_URL || "").trim();
	if (configured && !configured.includes("REPLACE_WITH")) return configured.replace(/\/+$/, "");
	return new URL(request.url).origin;
}

const record = (t) => ({
	ticket_id: t.ticket_id,
	customer_message: t.customer_message,
	customer_email: t.customer_email ?? null,
	status: t.status,
	created_at: t.created_at,
	messages: t.messages,
});

async function ticketByToken(env, token) {
	const ticket = token.length <= 64 ? await store.getByReplyToken(env, token) : null;
	if (!ticket) throw new HttpError(404, "الرابط غير صالح أو منتهي");
	return ticket;
}

async function route(request, env) {
	const url = new URL(request.url);
	const path = url.pathname;
	const method = request.method;

	if (path === "/health" && method === "GET") {
		await ensureSchema(env);
		return json({ status: "ok", ready: await isReady(env), runtime: "cloudflare-workers" });
	}

	if (path === "/login" && method === "POST") {
		await ensureSchema(env);
		const body = await readJson(request);
		const username = str(body, "username", { max: 100 });
		const password = str(body, "password", { max: 200 });
		if (await isLockedOut(env, username)) throw new HttpError(429, "محاولات دخول فاشلة كثيرة، حاول بعد قليل");
		const user = await authenticate(env, username, password);
		if (!user) {
			await recordFailedLogin(env, username);
			// رسالة عامة عمدًا (لا نوضح هل اسم المستخدم خطأ أم كلمة المرور).
			throw new HttpError(401, "اسم المستخدم أو كلمة المرور غير صحيحة");
		}
		await clearFailedLogins(env, username);
		return json({
			access_token: await createAccessToken(env, user),
			token_type: "bearer",
			role: user.role,
			display_name: user.display_name || user.username,
		});
	}

	if (path === "/admin/init" && method === "POST") {
		await requireEmployee(request, env);
		return json(await initStep(env));
	}

	if (path === "/draft" && method === "POST") {
		await requireEmployee(request, env);
		await ensureSchema(env);
		const body = await readJson(request);
		const ticketId = str(body, "ticket_id", { max: 64, required: false });
		const message = str(body, "customer_message", { min: 1, max: MAX_MESSAGE_CHARS, required: false });
		if (ticketId) {
			const ticket = await store.getTicket(env, ticketId);
			if (!ticket) throw new HttpError(404, "التذكرة غير موجودة");
			// جولة توضيح واحدة لكل تذكرة: بعد رد العميل لا نسأل مرة ثانية.
			return json(await handleRequest(env, store.conversationForRetrieval(ticket), !store.hadClarification(ticket)));
		}
		if (!message) throw new HttpError(422, "أرسل ticket_id أو customer_message");
		return json(await handleRequest(env, message));
	}

	if (path === "/submit-ticket" && method === "POST") {
		// نقطة الدخول الوحيدة للعميل: عامة عمدًا، ولا تُشغّل أي استرجاع أو توليد.
		await ensureSchema(env);
		const body = await readJson(request);
		const message = str(body, "customer_message", { min: 1, max: MAX_MESSAGE_CHARS });
		const email = str(body, "customer_email", { max: 254, required: false });
		if (email !== null && !EMAIL_RE.test(email)) throw new HttpError(422, "البريد الإلكتروني غير صالح");
		return json(record(await store.createTicket(env, message, email)));
	}

	if (path === "/tickets" && method === "GET") {
		await requireEmployee(request, env);
		await ensureSchema(env);
		return json((await store.listTickets(env, url.searchParams.get("status") || "pending")).map(record));
	}

	let m = path.match(/^\/tickets\/([^/]+)\/resolve$/);
	if (m && method === "POST") {
		const employee = await requireEmployee(request, env);
		await ensureSchema(env);
		const body = await readJson(request);
		const finalText = str(body, "final_text", { max: 10000 });
		const resolution = body.resolution;
		let ticket = await store.getTicket(env, m[1]);
		if (!ticket) throw new HttpError(404, "التذكرة غير موجودة");
		if (!["send", "clarify", "escalate"].includes(resolution)) throw new HttpError(422, "resolution يجب أن تكون 'send' أو 'clarify' أو 'escalate'");
		if (CLOSED_STATUSES.has(ticket.status)) throw new HttpError(409, "التذكرة مغلقة بالفعل (أُرسل الرد أو صُعّدت)");

		let emailSent = false;
		let emailError = null;
		let link = null;
		if (resolution === "send" || resolution === "clarify") {
			let text = finalText;
			if (resolution === "clarify") {
				link = `${publicUrl(env, request)}/app/reply.html?t=${ticket.reply_token}`;
				text = `${text}\n\nللرد على الأسئلة: ${link}`;
			}
			if (!ticket.customer_email) {
				emailError = "لا يوجد بريد إلكتروني مسجَّل لهذا العميل — أرسل الرد يدويًا.";
			} else {
				[emailSent, emailError] = await sendReplyEmail(
					env,
					ticket.customer_email,
					resolution === "clarify" ? "أسئلة بخصوص تذكرتك" : "رد بخصوص تذكرتك",
					text,
				);
			}
			// لا تتغير الحالة إلا لو الإيميل اتبعت فعلًا؛ فشل الإرسال يبقيها "pending".
			if (emailSent && resolution === "clarify") ticket = await store.startFollowUp(env, m[1], finalText);
			else if (emailSent) ticket = await store.closeTicket(env, m[1], "sent", finalText);
		} else {
			ticket = await store.closeTicket(env, m[1], "escalated");
		}
		console.log(`TICKET RESOLVED | id=${m[1]} | by=${employee.display_name} | resolution=${resolution} | email_sent=${emailSent}`);
		return json({ ticket: record(ticket), email_sent: emailSent, email_error: emailError, reply_link: link });
	}

	m = path.match(/^\/reply\/([^/]+)$/);
	if (m && method === "GET") {
		await ensureSchema(env);
		const ticket = await ticketByToken(env, m[1]);
		// لا يكشف أي بيانات غير أسئلة الموظف نفسها (لا إيميل ولا رقم تذكرة).
		const lastQ = [...ticket.messages].reverse().find((x) => x.kind === "clarification")?.text ?? "";
		return json({ questions: lastQ, can_reply: ticket.status === "awaiting_customer" });
	}
	if (m && method === "POST") {
		await ensureSchema(env);
		const ticket = await ticketByToken(env, m[1]);
		const body = await readJson(request);
		const message = str(body, "message", { min: 1, max: MAX_MESSAGE_CHARS });
		// الانتقال أولًا وذرّيًا: من ردّين متزامنين واحد فقط يغيّر الحالة ويُحفظ.
		if (!(await store.setStatus(env, ticket.ticket_id, "pending", new Set(["awaiting_customer"])))) {
			throw new HttpError(409, "تم استلام ردك بالفعل، وسيتواصل معك أحد موظفي الدعم.");
		}
		await store.addMessage(env, ticket.ticket_id, "customer", message);
		return json({ status: "received" });
	}

	if (path === "/" && method === "GET") return Response.redirect(new URL("/app/", request.url).toString(), 302);

	// الصفحات الثابتة تحت /app/ (مجلد stage4_production/static).
	if ((path === "/app" || path.startsWith("/app/")) && (method === "GET" || method === "HEAD")) {
		const assetPath = path === "/app" || path === "/app/" ? "/index.html" : path.slice(4);
		const res = await env.ASSETS.fetch(new Request(new URL(assetPath, request.url), request));
		return res.status === 404 ? json({ detail: "Not Found" }, 404) : res;
	}

	return json({ detail: "Not Found" }, 404);
}

export default {
	async fetch(request, env) {
		try {
			return await route(request, env);
		} catch (err) {
			if (err instanceof HttpError) {
				return json(typeof err.detail === "string" ? { detail: err.detail, ...err.extra } : { detail: err.detail }, err.status);
			}
			console.error("UNHANDLED", err?.stack ?? err);
			return json({ detail: "خطأ داخلي في الخادم" }, 500);
		}
	},
};
