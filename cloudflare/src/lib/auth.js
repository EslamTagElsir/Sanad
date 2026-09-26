// تسجيل دخول الموظفين: PBKDF2 (WebCrypto الأصلي) + JWT HS256.
// bcrypt غير مناسب لـ Workers (JS خالص، يتجاوز حد الـ CPU المجاني)، لذلك الصيغة:
//   pbkdf2-sha256$<iterations>$<salt b64>$<hash b64>
// لتوليد المستخدمين: node cloudflare/scripts/manage-users.mjs add <username> --name "..."
import { HttpError, b64decode, b64encode, b64urlDecode, b64urlEncode, enc, nowSeconds, timingSafeEqual } from "./util.js";

export const JWT_EXPIRY_SECONDS = 12 * 60 * 60;
export const VALID_ROLES = new Set(["employee", "customer"]);
const MAX_FAILED_LOGINS = 5;
const LOCKOUT_SECONDS = 5 * 60;
const MAX_PBKDF2_ITERATIONS = 100000; // حد Workers على PBKDF2

export async function hashPassword(plain, iterations = 10000, salt = crypto.getRandomValues(new Uint8Array(16))) {
	const key = await crypto.subtle.importKey("raw", enc.encode(plain), "PBKDF2", false, ["deriveBits"]);
	const bits = await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations }, key, 256);
	return `pbkdf2-sha256$${iterations}$${b64encode(salt)}$${b64encode(new Uint8Array(bits))}`;
}

export async function verifyPassword(plain, stored) {
	const parts = String(stored || "").split("$");
	if (parts.length !== 4 || parts[0] !== "pbkdf2-sha256") return false;
	const iterations = Number(parts[1]);
	if (!Number.isInteger(iterations) || iterations < 1 || iterations > MAX_PBKDF2_ITERATIONS) return false;
	const again = await hashPassword(plain, iterations, b64decode(parts[2]));
	return timingSafeEqual(again.split("$")[3], parts[3]);
}

let _usersRaw = null;
let _users = [];
export function loadUsers(env) {
	const raw = (env.SANAD_USERS_JSON || "").trim();
	if (raw !== _usersRaw) {
		_usersRaw = raw;
		try {
			_users = raw ? JSON.parse(raw) : [];
		} catch {
			_users = [];
		}
	}
	return _users;
}

export const findUser = (env, username) => loadUsers(env).find((u) => u.username === username) || null;

// وقت ثابت تقريبًا حتى لو المستخدم غير موجود (لا نكشف وجود اسم مستخدم).
const DUMMY_HASH = "pbkdf2-sha256$10000$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

export async function authenticate(env, username, password) {
	const user = findUser(env, username);
	const ok = await verifyPassword(password, user ? user.password_hash : DUMMY_HASH);
	return user && ok ? user : null;
}

// --- قفل مؤقت بعد محاولات فاشلة (في D1: يعمل عبر كل نسخ الـ Worker) ---
export async function isLockedOut(env, username) {
	const row = await env.DB.prepare("SELECT COUNT(*) AS n FROM login_failures WHERE username = ? AND ts > ?")
		.bind(username, nowSeconds() - LOCKOUT_SECONDS)
		.first();
	return (row?.n ?? 0) >= MAX_FAILED_LOGINS;
}

export async function recordFailedLogin(env, username) {
	await env.DB.batch([
		env.DB.prepare("DELETE FROM login_failures WHERE ts < ?").bind(nowSeconds() - LOCKOUT_SECONDS),
		env.DB.prepare("INSERT INTO login_failures (username, ts) VALUES (?, ?)").bind(username, nowSeconds()),
	]);
}

export const clearFailedLogins = (env, username) =>
	env.DB.prepare("DELETE FROM login_failures WHERE username = ?").bind(username).run();

// --- JWT HS256 ---
function secretOf(env) {
	const secret = env.SANAD_JWT_SECRET;
	if (!secret) throw new HttpError(503, "SANAD_JWT_SECRET غير مضبوط في أسرار الـ Worker");
	return secret;
}

const hmacKey = (secret, usages) =>
	crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, usages);

export async function createAccessToken(env, user) {
	const now = Math.floor(nowSeconds());
	const header = b64urlEncode(enc.encode(JSON.stringify({ alg: "HS256", typ: "JWT" })));
	const payload = b64urlEncode(
		enc.encode(
			JSON.stringify({
				sub: user.username,
				role: user.role,
				display_name: user.display_name || user.username,
				iat: now,
				exp: now + JWT_EXPIRY_SECONDS,
			}),
		),
	);
	const sig = await crypto.subtle.sign("HMAC", await hmacKey(secretOf(env), ["sign"]), enc.encode(`${header}.${payload}`));
	return `${header}.${payload}.${b64urlEncode(new Uint8Array(sig))}`;
}

// يتحقق من التوقيع والصلاحية ثم يطابق مع حساب المستخدم الحالي (حذف مستخدم أو تغيير
// دوره يسري فورًا؛ الدور يؤخذ من السجل الحالي لا من التوكن).
export async function decodeAccessToken(env, token) {
	const parts = String(token).split(".");
	if (parts.length !== 3) return null;
	try {
		const valid = await crypto.subtle.verify(
			"HMAC",
			await hmacKey(secretOf(env), ["verify"]),
			b64urlDecode(parts[2]),
			enc.encode(`${parts[0]}.${parts[1]}`),
		);
		if (!valid) return null;
		const header = JSON.parse(new TextDecoder().decode(b64urlDecode(parts[0])));
		if (header.alg !== "HS256") return null;
		const payload = JSON.parse(new TextDecoder().decode(b64urlDecode(parts[1])));
		for (const k of ["exp", "iat", "sub", "role"]) if (payload[k] === undefined) return null;
		if (payload.exp <= nowSeconds()) return null;
		const user = findUser(env, payload.sub);
		if (!user || !VALID_ROLES.has(user.role) || user.role !== payload.role) return null;
		return { ...payload, role: user.role, display_name: user.display_name || user.username };
	} catch (err) {
		if (err instanceof HttpError) throw err;
		return null;
	}
}

// --- تحديد هوية الطالب ---
function parseKeys(raw) {
	return String(raw || "").split(",").map((k) => k.trim()).filter(Boolean);
}
const keyIn = (candidate, keys) => keys.map((k) => timingSafeEqual(candidate, k)).some(Boolean);

// الترتيب: JWT ثم X-API-Key ثم وضع التطوير المفتوح (فقط إن فُعِّل صراحة). غير ذلك 401.
export async function getIdentity(request, env) {
	const authorization = request.headers.get("authorization");
	const apiKey = request.headers.get("x-api-key");
	if (authorization) {
		if (!authorization.startsWith("Bearer ")) throw new HttpError(401, "ترويسة Authorization غير صالحة");
		const payload = await decodeAccessToken(env, authorization.slice(7).trim());
		if (!payload) throw new HttpError(401, "جلسة الدخول منتهية أو غير صالحة، سجّل الدخول مرة أخرى");
		return payload;
	}
	if (apiKey) {
		if (keyIn(apiKey, parseKeys(env.SANAD_EMPLOYEE_KEYS))) return { sub: "api-key", role: "employee", display_name: "مفتاح API (موظف)" };
		if (keyIn(apiKey, parseKeys(env.SANAD_CUSTOMER_KEYS))) return { sub: "api-key", role: "customer", display_name: "مفتاح API (عميل)" };
		throw new HttpError(401, "مفتاح X-API-Key غير صالح");
	}
	if (["1", "true", "yes"].includes(String(env.SANAD_DEV_OPEN || "").toLowerCase())) {
		return { sub: "dev", role: "employee", display_name: "موظف (وضع تطوير)" };
	}
	throw new HttpError(401, "لازم تسجّل الدخول (Authorization: Bearer) أو ترويسة X-API-Key صالحة");
}

export async function requireEmployee(request, env) {
	const identity = await getIdentity(request, env);
	if (identity.role !== "employee") throw new HttpError(403, "هذا الطلب متاح لموظفي الدعم فقط");
	return identity;
}
