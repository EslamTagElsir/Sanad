// منطق الطلب الرئيسي: استرجاع -> ثقة معايَرة -> قرار (send_ready | needs_review | clarify | escalate)
// -> مسودة. يقابل handle_request في stage4_production/service.py.
import {
	FINAL_K,
	ESCALATE_BELOW,
	SEND_READY_ABOVE,
	embed,
	initStep,
	loadIndex,
	predictConfidence,
	retrieveWithSignals,
} from "./retrieval.js";
import { composeClarificationEmail, generateClarifyingQuestions, generateDraft, hasLlm, llmRerank, translateQuery } from "./llm.js";
import { HttpError, fnv1a } from "./util.js";

const draftCache = new Map();
const DRAFT_CACHE_MAX = 200;

export function escalationReason(confidence, hasSources, translated) {
	if (!hasSources) return "لا توجد مصادر في قاعدة المعرفة أو دليل السياسات أو التذاكر السابقة لهذه الرسالة.";
	const pct = Math.round(confidence * 100);
	if (!translated) {
		return (
			`الثقة منخفضة (${pct}%) لأن ترجمة رسالة العميل تعذّرت (خدمة الـ LLM لم تستجب)، ` +
			"والثقة بدون الترجمة تكون أقل عادةً لرسائل العامية. راجع المصادر أدناه — قد تكون مناسبة — " +
			"أو أعد توليد المسودة بعد قليل."
		);
	}
	return `الثقة منخفضة (${pct}%): المصادر المسترجعة غالبًا لا تجيب على رسالة العميل. راجعها أدناه، واكتب الرد يدويًا أو صعّد الحالة.`;
}

export function decideAction(confidence, top) {
	if (!top || confidence < ESCALATE_BELOW) return "escalate";
	if (confidence > SEND_READY_ABOVE && top.source_type === "kb") return "send_ready";
	return "needs_review";
}

/** الفهرس الجاهز، وإلا خطوة تجهيز واحدة ثم 503 (الواجهة تكرر /admin/init حتى الاكتمال). */
async function requireIndex(env) {
	const index = await loadIndex(env);
	if (index) return index;
	const progress = await initStep(env);
	if (progress.ready && (await loadIndex(env))) return loadIndex(env);
	throw new HttpError(503, "جاري تجهيز فهرس البحث لأول مرة، لحظات...", { init_pending: true, progress });
}

export async function handleRequest(env, customerMessage, allowClarify = true) {
	const started = Date.now();
	const key = fnv1a(`${allowClarify}::${customerMessage}`) + customerMessage.length;
	if (draftCache.has(key)) {
		return { ...draftCache.get(key), cached: true, latency_ms: Date.now() - started };
	}
	const index = await requireIndex(env);
	const timings = {};
	const timed = async (name, fn) => {
		const t = Date.now();
		try {
			return await fn();
		} finally {
			timings[name] = Date.now() - t;
		}
	};

	// الترجمة (LLM) وترميز الرسالة (Workers AI) بالتوازي؛ ترتيب المرشحين بالرسالة الأصلية فقط،
	// فترتيب الـ LLM مستقل عن الترجمة.
	const translationPromise = timed("translation", () => translateQuery(env, customerMessage));
	const [qVec] = await timed("embed", () => embed(env, [customerMessage], true));
	const first = retrieveWithSignals(index, qVec);
	const rerankPromise = first.candidates.length ? timed("rerank", () => llmRerank(env, customerMessage, first.candidates)) : Promise.resolve(null);
	const [translation, reranked] = await Promise.all([translationPromise, rerankPromise]);

	let translationVec = null;
	if (translation) [translationVec] = await timed("embed_translation", () => embed(env, [translation], true));
	const { candidates, signals } = retrieveWithSignals(index, qVec, translationVec);
	const top = candidates.length ? reranked || candidates.slice(0, FINAL_K) : [];

	// نموذج الثقة مُعايَر على تشابه أول نتيجة في ترتيب الـ embedding؛ ترتيب الـ LLM قد يقدّم
	// مصدرًا آخر، لذا نأخذ أعلى احتمال بين المصادر النهائية المعروضة.
	const confidence = Math.max(0, ...top.map((c) => predictConfidence(index.calibration, c, signals)));
	let action = decideAction(confidence, top[0]);

	const citations = top.map((c) => ({ title: c.title, source_type: c.source_type }));
	const translated = Boolean(translation);
	let questions = [];
	if (action === "escalate" && top.length && allowClarify) {
		const clar = await timed("clarify", () => generateClarifyingQuestions(env, customerMessage, top));
		if (clar && !clar.out_of_scope && clar.questions.length) {
			action = "clarify";
			questions = clar.questions;
		}
	}

	let draft, reason;
	if (action === "clarify") {
		draft = composeClarificationEmail(questions);
		reason =
			"الرسالة غير واضحة بما يكفي لاختيار الرد الصحيح. أرسل للعميل الأسئلة التوضيحية أدناه " +
			"(يمكنك تعديلها)، وسيصله رابط يرد منه على نفس التذكرة.";
	} else if (action === "escalate") {
		draft = "";
		reason = escalationReason(confidence, top.length > 0, translated);
	} else {
		draft = await timed("draft", () => generateDraft(env, customerMessage, top));
		reason = null;
	}

	const result = {
		draft,
		citations,
		confidence: Math.round(confidence * 1000) / 1000,
		action,
		reason,
		clarifying_questions: questions,
	};
	// نتيجة بدون ترجمة بسبب عطل مؤقت لا تُخزَّن: إعادة التوليد يجب أن تجرّب الترجمة من جديد.
	if (translated || !hasLlm(env)) {
		if (draftCache.size >= DRAFT_CACHE_MAX) for (const k of [...draftCache.keys()].slice(0, DRAFT_CACHE_MAX / 2)) draftCache.delete(k);
		draftCache.set(key, result);
	}
	const latency = Date.now() - started;
	console.log(`confidence=${confidence.toFixed(3)} | action=${action} | latency_ms=${latency} | timings=${JSON.stringify(timings)}`);
	return { ...result, cached: false, latency_ms: latency, timings };
}
