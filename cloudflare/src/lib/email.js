// إرسال الرد فعليًا للعميل عبر SendGrid (REST). بدون المفتاح: (false, رسالة) بدل استثناء،
// فيقدر الموظف يكمل يدويًا (نسخ المسودة وإرسالها بأي وسيلة).
export async function sendReplyEmail(env, toEmail, subject, bodyText) {
	if (!env.SENDGRID_API_KEY) return [false, "SENDGRID_API_KEY غير مضبوط — انسخ المسودة وأرسلها يدويًا."];
	if (!env.SENDGRID_FROM_EMAIL) return [false, "SENDGRID_FROM_EMAIL غير مضبوط."];
	try {
		const res = await fetch("https://api.sendgrid.com/v3/mail/send", {
			method: "POST",
			headers: { authorization: `Bearer ${env.SENDGRID_API_KEY}`, "content-type": "application/json" },
			body: JSON.stringify({
				personalizations: [{ to: [{ email: toEmail }] }],
				from: { email: env.SENDGRID_FROM_EMAIL },
				subject,
				content: [{ type: "text/plain", value: bodyText }],
			}),
			signal: AbortSignal.timeout(15000),
		});
		if ([200, 201, 202].includes(res.status)) return [true, null];
		return [false, `SendGrid رفض الإرسال (status=${res.status})`];
	} catch (err) {
		return [false, `فشل الاتصال بـ SendGrid: ${err?.message ?? err}`];
	}
}
