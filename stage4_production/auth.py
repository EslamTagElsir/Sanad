"""
stage4_production/auth.py — تسجيل دخول حقيقي للموظفين
---------------------------------------------------------
بديل عن مفتاح X-API-Key الثابت: تسجيل دخول باسم مستخدم/كلمة مرور، مع تجزئة
كلمات المرور (bcrypt) وتوكن جلسة (JWT) قصير الصلاحية. المستخدمون مخزّنون في
data/users.json (لا توجد قاعدة بيانات حقيقية في هذا المشروع التعليمي؛ لمشروع
إنتاج حقيقي استبدل هذا الملف بجدول مستخدمين في قاعدة بيانات فعلية).

إدارة المستخدمين: `python -m stage4_production.manage_users add`
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import bcrypt
import jwt

USERS_PATH = Path(__file__).parent.parent / "data" / "users.json"

# يُقرأ من AGENT_ASSIST_JWT_SECRET في .env. بدونه نولّد سرًا عشوائيًا لكل
# تشغيل — كافٍ للتطوير المحلي، لكنه يعني إبطال كل الجلسات عند كل إعادة تشغيل؛
# لإنتاج حقيقي اضبط قيمة ثابتة وسرية في .env دائمًا.
JWT_SECRET = os.environ.get("AGENT_ASSIST_JWT_SECRET")
if not JWT_SECRET:
    import secrets
    JWT_SECRET = secrets.token_hex(32)
    logging.getLogger("agent_copilot").warning("AGENT_ASSIST_JWT_SECRET غير مضبوط: سر مؤقت لهذا التشغيل فقط")
elif len(JWT_SECRET) < 32:
    logging.getLogger("agent_copilot").warning("AGENT_ASSIST_JWT_SECRET أقصر من 32 حرفًا — استخدم سرًا أطول")

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 12 * 60 * 60  # 12 ساعة
VALID_ROLES = {"employee", "customer"}

# قفل مؤقت بعد محاولات دخول فاشلة متكررة لنفس اسم المستخدم (ضد التخمين).
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 5 * 60
_failed_logins: dict[str, list[float]] = {}
_failed_lock = threading.Lock()


def load_users() -> list[dict]:
    if not USERS_PATH.exists():
        return []
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: list[dict]) -> None:
    USERS_PATH.parent.mkdir(exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def find_user(username: str) -> Optional[dict]:
    for u in load_users():
        if u["username"] == username:
            return u
    return None


def authenticate(username: str, password: str) -> Optional[dict]:
    user = find_user(username)
    if user is None or not verify_password(password, user["password_hash"]):
        return None
    return user


def _recent_failures(username: str) -> list[float]:
    cutoff = time.time() - LOCKOUT_SECONDS
    return [t for t in _failed_logins.get(username, []) if t > cutoff]


def is_locked_out(username: str) -> bool:
    with _failed_lock:
        return len(_recent_failures(username)) >= MAX_FAILED_LOGINS


def record_failed_login(username: str) -> None:
    with _failed_lock:
        _failed_logins[username] = _recent_failures(username) + [time.time()]


def clear_failed_logins(username: str) -> None:
    with _failed_lock:
        _failed_logins.pop(username, None)


def create_access_token(user: dict) -> str:
    now = int(time.time())
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "display_name": user.get("display_name", user["username"]),
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """يتحقق من التوقيع والصلاحية والحقول الإلزامية، ثم يطابق التوكن مع حساب
    المستخدم الحالي في users.json: حذف مستخدم أو تغيير دوره/أقسامه يسري فورًا
    على توكناته القائمة بدل انتظار انتهائها (12 ساعة). الدور يؤخذ من السجل
    الحالي، لا من التوكن."""
    try:
        payload = jwt.decode(
            token, JWT_SECRET, algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.PyJWTError:
        return None
    user = find_user(payload["sub"])
    if user is None or user.get("role") not in VALID_ROLES or user["role"] != payload["role"]:
        return None
    return {
        **payload,
        "role": user["role"],
        "display_name": user.get("display_name", user["username"]),
    }
