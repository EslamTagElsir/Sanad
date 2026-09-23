"""
stage4_production/email_service.py — إرسال الرد فعليًا للعميل عبر SendGrid
-------------------------------------------------------------------------
يعمل فقط إذا توفر SENDGRID_API_KEY في .env. بدونه، send_reply_email تعيد
(False, رسالة توضيحية) بدل رمي استثناء — الموظف يقدر يكمل يدويًا (نسخ
المسودة وإرسالها بأي وسيلة تانية) بدل ما الطلب كله يفشل.
"""

import os
from typing import Optional


def send_reply_email(to_email: str, subject: str, body_text: str) -> tuple[bool, Optional[str]]:
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        return False, "SENDGRID_API_KEY غير مضبوط في .env — انسخ المسودة وأرسلها يدويًا."

    from_email = os.environ.get("SENDGRID_FROM_EMAIL")
    if not from_email:
        return False, "SENDGRID_FROM_EMAIL غير مضبوط في .env."

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body_text,
    )
    try:
        client = SendGridAPIClient(api_key)
        response = client.send(message)
        if response.status_code in (200, 201, 202):
            return True, None
        return False, f"SendGrid رفض الإرسال (status={response.status_code})"
    except Exception as exc:  # واجهة SendGrid ترفع أنواع استثناءات متعددة لأخطاء الشبكة/التوثيق
        return False, f"فشل الاتصال بـ SendGrid: {exc}"
