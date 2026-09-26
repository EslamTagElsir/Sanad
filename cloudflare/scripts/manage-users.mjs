#!/usr/bin/env node
// إضافة موظف وتوليد قيمة السر SANAD_USERS_JSON (بديل manage_users.py: الـ Worker يستخدم PBKDF2
// لا bcrypt). كلمة المرور تُقرأ من متغير البيئة SANAD_NEW_PASSWORD أو تُطلب بلا إظهار.
//
//   SANAD_NEW_PASSWORD='كلمة-مرور-طويلة' node cloudflare/scripts/manage-users.mjs sara.ahmed "سارة أحمد"
//
// انسخ سطر JSON الناتج إلى Secret اسمه SANAD_USERS_JSON (Worker ← Settings ← Variables and secrets).
// لإضافة موظف آخر: مرّر الـ JSON الحالي في EXISTING_USERS_JSON وسيُضاف إليه.
import { webcrypto } from "node:crypto";
import readline from "node:readline";

const [username, displayName = username, role = "employee"] = process.argv.slice(2);
if (!username) {
	console.error('الاستخدام: node manage-users.mjs <username> ["اسم العرض"] [employee|customer]');
	process.exit(1);
}
// 10000 افتراضيًا: الخطة المجانية تسمح بـ 10ms CPU فقط لكل طلب (PBKDF2 الأصلي). على Workers Paid ارفعها (حتى 100000).
const ITERATIONS = Number(process.env.SANAD_PBKDF2_ITERATIONS) || 10000;

async function askPassword() {
	if (process.env.SANAD_NEW_PASSWORD) return process.env.SANAD_NEW_PASSWORD;
	const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
	return new Promise((resolve) => rl.question("كلمة المرور (10 أحرف على الأقل، ستظهر أثناء الكتابة): ", (a) => (rl.close(), resolve(a))));
}

const password = await askPassword();
if (password.length < 10) {
	console.error("كلمة المرور يجب ألا تقل عن 10 أحرف.");
	process.exit(1);
}
const salt = webcrypto.getRandomValues(new Uint8Array(16));
const key = await webcrypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
const bits = await webcrypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations: ITERATIONS }, key, 256);
const b64 = (u8) => Buffer.from(u8).toString("base64");
const users = JSON.parse(process.env.EXISTING_USERS_JSON || "[]").filter((u) => u.username !== username);
users.push({ username, display_name: displayName, role, password_hash: `pbkdf2-sha256$${ITERATIONS}$${b64(salt)}$${b64(new Uint8Array(bits))}` });
console.log("\nقيمة SANAD_USERS_JSON (سر — لا تنشرها):\n");
console.log(JSON.stringify(users));
