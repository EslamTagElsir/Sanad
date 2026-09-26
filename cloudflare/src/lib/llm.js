// كل مهام الـ LLM (ترجمة السؤال، ترتيب المصادر، أسئلة توضيحية، المسودة) عبر OpenRouter.
// نفس منطق common.py: بدون مفتاح أو عند الفشل يرجع null والمستدعي يكمل بسلوك احتياطي.
import prompts from "../data/prompts.json" with { type: "json" };

const DEFAULT_MODELS = ["inclusionai/ling-3.0-flash-fin:free"];
const TIMEOUT_MS = 20000;
const SENSITIVE_RE = new RegExp(prompts.sensitive_regex, "i");
const SOURCE_KIND = prompts.source_kind;
const MAX_Q = prompts.max_clarifying_questions;

// حجب مؤقت بعد نفاد الحصة اليومية (يوفّر انتظار الرفض في كل طلب).
let quotaBlockedUntil = 0;

export const hasLlm = (env) => Boolean(env.OPENROUTER_API_KEY);

function modelsOf(env) {
	const list = String(env.OPENROUTER_MODELS || "").split(",").map((m) => m.trim()).filter(Boolean);
	return list.length ? list : DEFAULT_MODELS;
}

async function callOnce(env, models, messages, maxTokens, reasoning) {
	const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
		method: "POST",
		headers: { authorization: `Bearer ${env.OPENROUTER_API_KEY}`, "content-type": "application/json" },
		body: JSON.stringify({ model: models[0], models, messages, max_tokens: maxTokens, reasoning }),
		signal: AbortSignal.timeout(TIMEOUT_MS),
	});
	const text = await res.text();
	if (!res.ok) {
		const err = new Error(`OpenRouter ${res.status}: ${text.slice(0, 300)}`);
		err.status = res.status;
		err.body = text;
		err.headers = res.headers;
		throw err;
	}
	return JSON.parse(text);
}

export async function openrouterChat(env, messages, maxTokens, task) {
	if (!hasLlm(env) || Date.now() < quotaBlockedUntil) return null;
	const models = modelsOf(env);
	const started = Date.now();
	try {
		let data;
		try {
			data = await callOnce(env, models, messages, maxTokens, { enabled: false });
		} catch (err) {
			// بعض الموديلات تفرض التفكير وترفض إيقافه — نعيد بأقل تفكير ممكن.
			if (err.status === 400 && /reasoning is mandatory/i.test(err.body || "")) {
				data = await callOnce(env, models, messages, maxTokens, { effort: "low", exclude: true });
			} else throw err;
		}
		const content = String(data?.choices?.[0]?.message?.content ?? "").trim();
		console.log(`OPENROUTER | task=${task} | model=${data?.model} | ${Date.now() - started}ms | chars=${content.length}`);
		return content || null;
	} catch (err) {
		if (err.status === 429 && /per-day/i.test(err.body || "")) {
			const reset = Number(err.headers?.get?.("x-ratelimit-reset"));
			quotaBlockedUntil = reset > 1e12 ? reset : Date.now() + 3600_000;
		}
		console.warn(`OPENROUTER | task=${task} | failed after ${Date.now() - started}ms | ${err?.message ?? err}`);
		return null;
	}
}

// --- ترجمة السؤال (عربي<->إنجليزي) ---
const translationCache = new Map();
export async function translateQuery(env, question) {
	const key = question.trim();
	if (translationCache.has(key)) return translationCache.get(key);
	const prompt =
		`ترجم النص التالي إلى العربية إن كان إنجليزيًا، أو إلى الإنجليزية إن كان عربيًا. ` +
		`أعد الترجمة فقط بدون أي شرح إضافي:\n\n${question}`;
	const translation = await openrouterChat(env, [{ role: "user", content: prompt }], 200, "translate");
	if (translation) {
		if (translationCache.size >= 500) for (const k of [...translationCache.keys()].slice(0, 250)) translationCache.delete(k);
		translationCache.set(key, translation);
	}
	return translation;
}

// --- ترتيب المصادر بحكم LLM ---
export async function llmRerank(env, question, candidates, topK = 4) {
	if (!candidates.length) return null;
	const blocks = candidates.map((c, i) => {
		const kind = { kb: "مقالة KB رسمية", manual: "دليل السياسات" }[c.source_type] ?? "تذكرة سابقة";
		return `${i + 1}. [${kind}] ${c.title}\n${c.text.slice(0, 200)}`;
	});
	const prompt =
		`رسالة العميل: ${question}\n\nالمصادر المرشحة:\n${blocks.join("\n\n")}` +
		"\n\nرتّب أرقام المصادر من الأكثر صلة فعليًا بمضمون رسالة العميل إلى الأقل. " +
		"استبعد تمامًا أي رقم مصدره غير ذي صلة حقيقية بالسؤال. " +
		"أعد الأرقام فقط مفصولة بفواصل بدون أي شرح إضافي، مثال: 3,1,4";
	const raw = await openrouterChat(env, [{ role: "user", content: prompt }], 50, "rerank");
	if (raw === null) return null;
	const seen = [];
	for (const tok of raw.match(/\d+/g) ?? []) {
		const idx = Number(tok) - 1;
		if (idx >= 0 && idx < candidates.length && !seen.includes(idx)) seen.push(idx);
	}
	return seen.length ? seen.slice(0, topK).map((i) => candidates[i]) : null;
}

// --- المسودة ---
function buildPrompt(question, chunks) {
	const blocks = chunks.map((c) => `[${SOURCE_KIND[c.source_type] ?? SOURCE_KIND.past_ticket}: ${c.title}]\n${c.text}`);
	return `السياق:\n${blocks.join("\n\n---\n\n")}\n\nرسالة العميل: ${question}\n\nمسودة الرد للموظف:`;
}

export async function generateDraft(env, question, chunks) {
	if (!chunks.length) return "لا توجد معلومات كافية في قاعدة المعرفة، يُنصح بتصعيد الحالة.";
	const content = await openrouterChat(
		env,
		[
			{ role: "system", content: prompts.system_prompt },
			{ role: "user", content: buildPrompt(question, chunks) },
		],
		700,
		"draft",
	);
	if (content) return content;
	// وضع بدون API: مسودة استخراجية بسيطة.
	const top = chunks[0];
	let kind = SOURCE_KIND[top.source_type] ?? SOURCE_KIND.past_ticket;
	if (top.source_type === "past_ticket") kind += "، تحقق قبل الإرسال";
	return `[وضع بدون مفتاح API — مسودة استخراجية]\nأقرب مصدر (${kind}): ${top.title}\n"${top.text}"`;
}

// --- أسئلة توضيحية ---
export function parseClarification(content) {
	if (!content) return null;
	let data = null;
	const match = content.match(/\{[\s\S]*\}/);
	if (match) {
		try {
			data = JSON.parse(match[0]);
		} catch {
			data = null;
		}
	}
	let outOfScope = false;
	let raw;
	if (data && typeof data === "object" && !Array.isArray(data)) {
		outOfScope = Boolean(data.out_of_scope);
		raw = (Array.isArray(data.questions) ? data.questions : []).filter((q) => typeof q === "string");
	} else {
		raw = content.split("\n").filter((l) => /[?؟]\s*$/.test(l.trim()));
	}
	const questions = [];
	for (let q of raw) {
		q = q.replace(/^\s*(\d+[.)-]|[-*•])\s*/, "").trim();
		if (q && !SENSITIVE_RE.test(q) && !questions.includes(q)) questions.push(q);
	}
	if (outOfScope) return { out_of_scope: true, questions: [] };
	const limited = questions.slice(0, MAX_Q);
	return limited.length ? { out_of_scope: false, questions: limited } : null;
}

export async function generateClarifyingQuestions(env, question, chunks) {
	const topics = chunks.map((c) => `- ${c.title}`).join("\n") || "- (لا توجد)";
	const prompt = prompts.clarify_prompt.replace("%%TOPICS%%", topics).replace("%%QUESTION%%", question);
	return parseClarification(await openrouterChat(env, [{ role: "user", content: prompt }], 300, "clarify"));
}

export function composeClarificationEmail(questions) {
	const lines = questions.map((q, i) => `${i + 1}. ${q}`).join("\n");
	return (
		"أهلًا بحضرتك، شكرًا لتواصلك معنا.\n" +
		"عشان نقدر نساعدك بدقة، محتاجين نعرف شوية تفاصيل:\n\n" +
		`${lines}\n\n` +
		"ممكن ترد على الأسئلة دي من الرابط اللي تحت، وهنكمل معاك على طول."
	);
}
