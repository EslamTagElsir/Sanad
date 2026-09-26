// أدوات صغيرة مشتركة (بدون أي اعتماديات خارجية: Worker على الخطة المجانية CPU 10ms).
const enc = new TextEncoder();
export { enc };

export class HttpError extends Error {
	constructor(status, detail, extra = {}) {
		super(typeof detail === "string" ? detail : JSON.stringify(detail));
		this.status = status;
		this.detail = detail;
		this.extra = extra;
	}
}

export function json(data, status = 200) {
	return new Response(JSON.stringify(data), {
		status,
		headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
	});
}

export const nowSeconds = () => Date.now() / 1000;

export function b64encode(bytes) {
	let s = "";
	for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
	return btoa(s);
}

export function b64decode(str) {
	const bin = atob(str);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}

export const b64urlEncode = (bytes) => b64encode(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

export function b64urlDecode(s) {
	s = s.replace(/-/g, "+").replace(/_/g, "/");
	while (s.length % 4) s += "=";
	return b64decode(s);
}

export function randomHex(nBytes) {
	return [...crypto.getRandomValues(new Uint8Array(nBytes))].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export const randomUrlSafe = (nBytes) => b64urlEncode(crypto.getRandomValues(new Uint8Array(nBytes)));

// مقارنة ثابتة الزمن (لا تسريب لمحتوى المفتاح عبر توقيت المقارنة).
export function timingSafeEqual(a, b) {
	const ea = enc.encode(String(a));
	const eb = enc.encode(String(b));
	let diff = ea.length ^ eb.length;
	const len = Math.max(ea.length, eb.length);
	for (let i = 0; i < len; i++) diff |= (ea[i] ?? 0) ^ (eb[i] ?? 0);
	return diff === 0;
}

// FNV-1a 32-bit: بصمة سريعة لتغيّر المصادر (ليست للأمان).
export function fnv1a(str, seed = 0x811c9dc5) {
	let h = seed >>> 0;
	for (let i = 0; i < str.length; i++) {
		h ^= str.charCodeAt(i);
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h.toString(16).padStart(8, "0");
}
