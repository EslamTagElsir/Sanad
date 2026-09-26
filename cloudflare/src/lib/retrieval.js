// الاسترجاع الدلالي على Workers AI (@cf/qwen/qwen3-embedding-0.6b — نفس نموذج النسخة
// الأصلية Qwen3-Embedding-0.6B) + ثقة معايَرة (Logistic Regression على تشابه أول نتيجة).
//
// قيود الخطة المجانية: 10ms CPU لكل طلب، لذلك التجهيز الأولي (ترميز القطع، ترميز
// المجموعة الذهبية، حساب معايرة الثقة) يجري على خطوات صغيرة قابلة للاستئناف
// (initStep) وتُحفظ نتيجتها في D1؛ بعدها كل طلب عادي خفيف.
import chunks from "../data/chunks.json" with { type: "json" };
import goldenSet from "../data/golden_set.json" with { type: "json" };
import goldenTranslations from "../data/golden_translations.json" with { type: "json" };
import { ensureSchema } from "./db.js";
import { b64decode, b64encode, fnv1a } from "./util.js";

export const EMBED_MODEL = "@cf/qwen/qwen3-embedding-0.6b";
const QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";
const EMBED_BATCH = 16;
const CHUNKS_PER_STEP = 32;
const GOLDEN_PER_STEP = 32;
const CALIB_PER_STEP = 20;

export const CANDIDATE_POOL = 8;
export const FINAL_K = 4;
export const ESCALATE_BELOW = 0.4;
export const SEND_READY_ABOVE = 0.8;

const golden = goldenSet.items;

// ---------------------------------------------------------------------------
// Workers AI
// ---------------------------------------------------------------------------
function normalize(v) {
	let n = 0;
	for (let i = 0; i < v.length; i++) n += v[i] * v[i];
	n = Math.sqrt(n) || 1;
	const out = new Float32Array(v.length);
	for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
	return out;
}

export async function embed(env, texts, isQuery = false) {
	const out = [];
	for (let i = 0; i < texts.length; i += EMBED_BATCH) {
		const batch = texts.slice(i, i + EMBED_BATCH).map((t) => (isQuery ? QUERY_PREFIX + t : t));
		const res = await env.AI.run(EMBED_MODEL, { text: batch });
		const data = res?.data ?? res?.embeddings ?? res?.result?.data;
		if (!Array.isArray(data) || data.length !== batch.length) throw new Error("رد Workers AI للـ embedding غير متوقع");
		for (const v of data) out.push(normalize(v));
	}
	return out;
}

// ---------------------------------------------------------------------------
// تخزين المتجهات في D1 (int8 مع معامل تحجيم لكل متجه، ~1.4KB لكل قطعة)
// ---------------------------------------------------------------------------
function quantize(vec) {
	let max = 0;
	for (let i = 0; i < vec.length; i++) max = Math.max(max, Math.abs(vec[i]));
	const scale = max / 127 || 1;
	const q = new Int8Array(vec.length);
	for (let i = 0; i < vec.length; i++) q[i] = Math.round(vec[i] / scale);
	return { scale, vec: b64encode(new Uint8Array(q.buffer)) };
}

function dequantize(scale, b64, into, offset) {
	const q = new Int8Array(b64decode(b64).buffer);
	for (let i = 0; i < q.length; i++) into[offset + i] = q[i] * scale;
	return q.length;
}

const insertEmbedding = (env, id, kind, vec) => {
	const { scale, vec: b64 } = quantize(vec);
	return env.DB.prepare("INSERT OR REPLACE INTO embeddings (id, kind, scale, vec) VALUES (?, ?, ?, ?)").bind(id, kind, scale, b64);
};

// ---------------------------------------------------------------------------
// بصمات المصادر: تغيّرها يعيد التجهيز تلقائيًا
// ---------------------------------------------------------------------------
const chunkFingerprint = fnv1a(EMBED_MODEL + "\n" + chunks.map((c) => c.chunk_id + "\u0001" + c.embed_text).join("\u0002"));
const goldenTexts = () =>
	golden.flatMap((g) => {
		const t = goldenTranslations[g.id];
		return [{ id: "gq:" + g.id, text: g.question }, ...(t ? [{ id: "gt:" + g.id, text: t }] : [])];
	});
const goldenFingerprint = fnv1a(chunkFingerprint + JSON.stringify(golden) + JSON.stringify(goldenTranslations));

async function getMeta(env) {
	const rows = await env.DB.prepare("SELECT key, value FROM meta").all();
	return Object.fromEntries((rows.results ?? []).map((r) => [r.key, r.value]));
}
const setMeta = (env, key, value) => env.DB.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)").bind(key, value);

// ---------------------------------------------------------------------------
// الفهرس في الذاكرة (لكل isolate)
// ---------------------------------------------------------------------------
let _index = null; // { matrix, dim, calibration, key }

async function loadMatrix(env) {
	const rows = (await env.DB.prepare("SELECT id, scale, vec FROM embeddings WHERE kind = 'chunk'").all()).results ?? [];
	if (rows.length < chunks.length) return null;
	const byId = new Map(rows.map((r) => [r.id, r]));
	let matrix = null;
	let dim = 0;
	for (let i = 0; i < chunks.length; i++) {
		const row = byId.get(chunks[i].chunk_id);
		if (!row) return null;
		if (!matrix) {
			dim = b64decode(row.vec).length;
			matrix = new Float32Array(chunks.length * dim);
		}
		dequantize(row.scale, row.vec, matrix, i * dim);
	}
	return { matrix, dim };
}

/** الفهرس الجاهز، أو null إن لم يكتمل التجهيز. */
export async function loadIndex(env) {
	await ensureSchema(env);
	if (_index && _index.key === goldenFingerprint) return _index;
	const meta = await getMeta(env);
	if (meta.chunk_fp !== chunkFingerprint || meta.golden_fp !== goldenFingerprint || !meta.calibration) return null;
	const m = await loadMatrix(env);
	if (!m) return null;
	_index = { ...m, calibration: JSON.parse(meta.calibration), key: goldenFingerprint };
	return _index;
}

export async function isReady(env) {
	return (await loadIndex(env)) !== null;
}

// ---------------------------------------------------------------------------
// البحث
// ---------------------------------------------------------------------------
function similarities(index, qVec) {
	const { matrix, dim } = index;
	const n = matrix.length / dim;
	const sims = new Float32Array(n);
	for (let r = 0; r < n; r++) {
		let s = 0;
		const off = r * dim;
		for (let i = 0; i < dim; i++) s += matrix[off + i] * qVec[i];
		sims[r] = s;
	}
	return sims;
}

/**
 * الترتيب بتشابه الرسالة الأصلية. الإشارة (dense): أعلى تشابه بين الأصل وترجمته
 * (translationVec اختياري). يرجع { candidates, signals } كما في retrieve_with_signals.
 */
export function retrieveWithSignals(index, qVec, translationVec = null, pool = CANDIDATE_POOL) {
	const simsQ = similarities(index, qVec);
	const simsT = translationVec ? similarities(index, translationVec) : null;
	const order = [...simsQ.keys()].sort((a, b) => simsQ[b] - simsQ[a]);
	const signals = {};
	for (const i of order) signals[chunks[i].chunk_id] = { dense: simsT ? Math.max(simsQ[i], simsT[i]) : simsQ[i] };
	return { candidates: order.slice(0, Math.min(pool, chunks.length)).map((i) => chunks[i]), signals };
}

// ---------------------------------------------------------------------------
// الثقة المعايَرة: StandardScaler + LogisticRegression(C=1) على ميزة واحدة (dense)
// ---------------------------------------------------------------------------
export function fitCalibration(rows) {
	const y = rows.map((r) => r.label);
	if (new Set(y).size < 2) return { mode: "raw" };
	const xs = rows.map((r) => r.dense);
	const n = xs.length;
	const mean = xs.reduce((a, b) => a + b, 0) / n;
	const std = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
	const z = xs.map((x) => (x - mean) / std);
	let w = 0;
	let b = 0;
	for (let iter = 0; iter < 100; iter++) {
		let gw = w; // مشتقة الجزء المنتظَم 0.5*w^2/C مع C=1
		let gb = 0;
		let hww = 1;
		let hwb = 0;
		let hbb = 0;
		for (let i = 0; i < n; i++) {
			const p = 1 / (1 + Math.exp(-(w * z[i] + b)));
			const e = p - y[i];
			const v = p * (1 - p);
			gw += e * z[i];
			gb += e;
			hww += v * z[i] * z[i];
			hwb += v * z[i];
			hbb += v;
		}
		const det = hww * hbb - hwb * hwb || 1e-12;
		const dw = (hbb * gw - hwb * gb) / det;
		const db = (hww * gb - hwb * gw) / det;
		w -= dw;
		b -= db;
		if (Math.abs(dw) + Math.abs(db) < 1e-9) break;
	}
	return { mode: "lr", mean, std, w, b, n, positives: y.reduce((a, c) => a + c, 0) };
}

export function predictConfidence(calibration, chunk, signals) {
	if (!chunk || !signals[chunk.chunk_id]) return 0;
	const x = signals[chunk.chunk_id].dense;
	if (!calibration || calibration.mode !== "lr") return Math.min(Math.max(x, 0), 1);
	return 1 / (1 + Math.exp(-(calibration.w * ((x - calibration.mean) / calibration.std) + calibration.b)));
}

// المصدر صحيح إذا كان ضمن المتوقع، أو قطعة من دليل السياسات تشرح نفس السياسة (refs ≤ 3).
export function isCorrectSource(chunk, expected) {
	if (expected.includes(chunk.source_id)) return true;
	const refs = chunk.refs ?? [];
	return refs.length > 0 && refs.length <= 3 && refs.some((r) => expected.includes(r));
}

// ---------------------------------------------------------------------------
// التجهيز على خطوات (كل استدعاء = خطوة صغيرة واحدة)
// ---------------------------------------------------------------------------
/** يُرجع { ready, stage, done, total } ويحفظ التقدم في D1. */
export async function initStep(env) {
	await ensureSchema(env);
	const meta = await getMeta(env);

	if (meta.chunk_fp !== chunkFingerprint) {
		await env.DB.batch([
			env.DB.prepare("DELETE FROM embeddings"),
			env.DB.prepare("DELETE FROM calib_rows"),
			env.DB.prepare("DELETE FROM meta WHERE key IN ('calibration', 'golden_fp')"),
			setMeta(env, "chunk_fp", chunkFingerprint),
		]);
		meta.chunk_fp = chunkFingerprint;
		delete meta.calibration;
		delete meta.golden_fp;
	} else if (meta.golden_fp && meta.golden_fp !== goldenFingerprint) {
		await env.DB.batch([
			env.DB.prepare("DELETE FROM embeddings WHERE kind != 'chunk'"),
			env.DB.prepare("DELETE FROM calib_rows"),
			env.DB.prepare("DELETE FROM meta WHERE key IN ('calibration', 'golden_fp')"),
		]);
		delete meta.calibration;
		delete meta.golden_fp;
	}
	if (meta.calibration && meta.golden_fp === goldenFingerprint) return { ready: true, stage: "ready", done: 1, total: 1 };

	const have = new Set(((await env.DB.prepare("SELECT id FROM embeddings").all()).results ?? []).map((r) => r.id));

	// 1) ترميز قطع قاعدة المعرفة ودليل السياسات والتذاكر السابقة
	const missingChunks = chunks.filter((c) => !have.has(c.chunk_id));
	if (missingChunks.length) {
		const slice = missingChunks.slice(0, CHUNKS_PER_STEP);
		const vecs = await embed(env, slice.map((c) => c.embed_text), false);
		await env.DB.batch(slice.map((c, i) => insertEmbedding(env, c.chunk_id, "chunk", vecs[i])));
		return { ready: false, stage: "chunks", done: chunks.length - missingChunks.length + slice.length, total: chunks.length };
	}

	// 2) ترميز أسئلة المجموعة الذهبية وترجماتها (كاستعلامات)
	const gTexts = goldenTexts();
	const missingGolden = gTexts.filter((g) => !have.has(g.id));
	if (missingGolden.length) {
		const slice = missingGolden.slice(0, GOLDEN_PER_STEP);
		const vecs = await embed(env, slice.map((g) => g.text), true);
		await env.DB.batch(slice.map((g, i) => insertEmbedding(env, g.id, "golden", vecs[i])));
		return { ready: false, stage: "golden", done: gTexts.length - missingGolden.length + slice.length, total: gTexts.length };
	}

	// 3) ميزة الثقة لكل سؤال ذهبي (أعلى نتيجة + تسميتها)، ثم تدريب النموذج
	const doneRows = new Set(((await env.DB.prepare("SELECT item_id FROM calib_rows").all()).results ?? []).map((r) => r.item_id));
	const todo = golden.filter((g) => !doneRows.has(g.id));
	if (todo.length) {
		const m = await loadMatrix(env);
		if (!m) throw new Error("متجهات القطع ناقصة");
		const vecRows = (await env.DB.prepare("SELECT id, scale, vec FROM embeddings WHERE kind = 'golden'").all()).results ?? [];
		const vecOf = new Map(vecRows.map((r) => [r.id, r]));
		const load = (id) => {
			const r = vecOf.get(id);
			if (!r) return null;
			const out = new Float32Array(m.dim);
			dequantize(r.scale, r.vec, out, 0);
			return out;
		};
		const slice = todo.slice(0, CALIB_PER_STEP);
		const stmts = [];
		for (const g of slice) {
			const { candidates, signals } = retrieveWithSignals(m, load("gq:" + g.id), load("gt:" + g.id));
			const top = candidates[0];
			const label = g.in_scope && isCorrectSource(top, g.expected_sources) ? 1 : 0;
			stmts.push(env.DB.prepare("INSERT OR REPLACE INTO calib_rows (item_id, dense, label) VALUES (?, ?, ?)").bind(g.id, signals[top.chunk_id].dense, label));
		}
		await env.DB.batch(stmts);
		if (todo.length > slice.length) {
			return { ready: false, stage: "calibration", done: golden.length - todo.length + slice.length, total: golden.length };
		}
	}
	const rows = (await env.DB.prepare("SELECT dense, label FROM calib_rows").all()).results ?? [];
	const calibration = fitCalibration(rows);
	await env.DB.batch([setMeta(env, "calibration", JSON.stringify(calibration)), setMeta(env, "golden_fp", goldenFingerprint)]);
	_index = null;
	return { ready: true, stage: "ready", done: 1, total: 1 };
}

export const TOTAL_CHUNKS = chunks.length;
