"""
stage4_production/manage_users.py — إضافة موظفين
-------------------------------------------------
    python -m stage4_production.manage_users add sara.ahmed --name "سارة أحمد"

يطلب كلمة المرور (لا تظهر أثناء الكتابة)، يخزّنها مجزّأة (bcrypt) في
data/users.json، ثم يطبع محتوى الملف كسطر JSON واحد لتضعه في الإعداد السري
SANAD_USERS_JSON على السيرفر (Hugging Face Spaces ← Settings ← Secrets)، لأن
users.json نفسه لا يُرفع إلى GitHub.
"""

import argparse
import getpass
import json
import sys

from stage4_production import auth


def add(username: str, display_name: str, role: str) -> None:
    users = auth.load_users()
    if any(u["username"] == username for u in users):
        sys.exit(f"المستخدم {username} موجود بالفعل.")
    password = getpass.getpass("كلمة المرور: ")
    if len(password) < 10:
        sys.exit("كلمة المرور يجب ألا تقل عن 10 أحرف.")
    if password != getpass.getpass("تأكيد كلمة المرور: "):
        sys.exit("كلمتا المرور غير متطابقتين.")
    users.append({"username": username, "display_name": display_name or username,
                  "role": role, "password_hash": auth.hash_password(password)})
    auth.save_users(users)
    print(f"تمت إضافة {username} إلى {auth.USERS_PATH}.")
    print("\nقيمة SANAD_USERS_JSON للسيرفر (سر — لا تنشرها):")
    print(json.dumps(users, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="إدارة موظفي Sanad")
    sub = parser.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add", help="إضافة موظف")
    p_add.add_argument("username")
    p_add.add_argument("--name", default="", help="اسم العرض")
    p_add.add_argument("--role", default="employee", choices=sorted(auth.VALID_ROLES))
    args = parser.parse_args()
    if args.command == "add":
        add(args.username, args.name, args.role)


if __name__ == "__main__":
    main()
