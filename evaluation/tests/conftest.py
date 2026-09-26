"""
إعداد مشترك للاختبارات: بيئة معزولة وحتمية بالكامل.
  - كل مفاتيح المزوّدين الخارجيين فارغة (قبل أي import لـ common الذي يحمّل
    .env؛ load_dotenv لا يستبدل متغيرًا موجودًا) → لا استدعاءات LLM ولا إيميل.
  - نموذج الـ embedding الافتراضي الحقيقي (Qwen3-Embedding، من الكاش المحلي)
    لأن الاسترجاع كله قائم عليه.
  - فهرس ومستخدمون في مجلد مؤقت، لا يلمسون data/users.json ولا store/.
"""

import os
import sys
from pathlib import Path

for var in ("OPENROUTER_API_KEY", "SENDGRID_API_KEY",
            "SANAD_EMPLOYEE_KEYS", "SANAD_CUSTOMER_KEYS", "SANAD_DEV_OPEN",
            "SANAD_EMBED_MODEL"):
    os.environ[var] = ""
os.environ["SANAD_JWT_SECRET"] = "test-secret-" + "x" * 40

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import pytest
from fastapi.testclient import TestClient

from stage2_hybrid import rag, confidence
from stage4_production import auth, service, ticket_store

PASSWORD = "correct horse battery staple"
USERS = [
    {"username": "sara.ahmed", "display_name": "سارة", "role": "employee"},
    {"username": "cust", "display_name": "عميل", "role": "customer"},
]


@pytest.fixture(scope="session", autouse=True)
def isolated_index(tmp_path_factory):
    store = tmp_path_factory.mktemp("store")
    rag.STORE_DIR = store
    rag.INDEX_PATH = store / "index_dense.pkl"
    rag._INDEX = None
    confidence.reset_model()
    rag.build_index()
    yield


@pytest.fixture(scope="session")
def users_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("users") / "users.json"
    pw_hash = auth.hash_password(PASSWORD)
    path.write_text(json.dumps([{**u, "password_hash": pw_hash} for u in USERS], ensure_ascii=False), encoding="utf-8")
    return path


def _cached_translation(question: str):
    """بديل حتمي لترجمة الـ LLM: الترجمة المحفوظة لأسئلة المجموعة الذهبية، وNone
    لغيرها (نفس سلوك الخدمة حين لا يتوفر مزوّد LLM)."""
    by_question = {it["question"]: confidence.load_golden_translations().get(it["id"])
                   for it in confidence.load_golden()}
    return by_question.get(question)


@pytest.fixture(autouse=True)
def isolated_state(users_file, monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "translate_query_for_retrieval", _cached_translation)
    monkeypatch.setattr(auth, "USERS_PATH", users_file)
    monkeypatch.setattr(service, "EMPLOYEE_KEYS", set())
    monkeypatch.setattr(service, "CUSTOMER_KEYS", set())
    monkeypatch.setattr(service, "DEV_OPEN", False)
    service._draft_cache.clear()
    monkeypatch.setattr(ticket_store, "DB_PATH", tmp_path / "tickets.db")   # قاعدة تذاكر نظيفة لكل اختبار
    monkeypatch.setattr(ticket_store, "FOLLOWUP_DB_PATH", tmp_path / "followups.db")
    monkeypatch.setattr(ticket_store, "D1_URL", "")
    auth._failed_logins.clear()
    yield


@pytest.fixture(scope="session")
def client():
    with TestClient(service.app) as c:
        yield c


def token_for(username: str) -> str:
    user = next(u for u in USERS if u["username"] == username)
    return auth.create_access_token(user)


def bearer(username: str) -> dict:
    return {"Authorization": f"Bearer {token_for(username)}"}
