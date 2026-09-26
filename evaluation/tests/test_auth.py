"""اختبارات المصادقة — أهمها انحدار ثغرة التجاوز: غياب مفاتيح X-API-Key في
.env كان يجعل أي طلب مجهول يُعامَل كموظف."""

import time

import jwt
import pytest

from stage4_production import auth, service
from evaluation.tests.conftest import PASSWORD, bearer

PROTECTED = [
    ("post", "/draft", {"customer_message": "حسابي اتجمد"}),
    ("get", "/tickets", None),
    ("post", "/tickets/abc/resolve", {"final_text": "x", "resolution": "escalate"}),
]


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_anonymous_request_is_rejected_even_without_api_keys(client, method, path, body):
    assert not service.EMPLOYEE_KEYS and not service.CUSTOMER_KEYS
    resp = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert resp.status_code == 401


def test_dev_open_mode_is_off_by_default():
    import os
    assert os.environ.get("SANAD_DEV_OPEN") == ""
    assert service.DEV_OPEN is False


def test_dev_open_mode_only_when_explicitly_enabled(client, monkeypatch):
    monkeypatch.setattr(service, "DEV_OPEN", True)
    assert client.get("/tickets").status_code == 200


@pytest.mark.parametrize("header", ["Bearer not-a-jwt", "Basic dXNlcjpwYXNz", "Bearer "])
def test_invalid_authorization_header_is_rejected(client, header):
    assert client.get("/tickets", headers={"Authorization": header}).status_code == 401


def test_token_signed_with_other_secret_is_rejected(client):
    forged = jwt.encode({"sub": "sara.ahmed", "role": "employee", "iat": int(time.time()),
                         "exp": int(time.time()) + 60}, "attacker-secret", algorithm="HS256")
    assert client.get("/tickets", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_token_with_alg_none_is_rejected(client):
    forged = jwt.encode({"sub": "sara.ahmed", "role": "employee", "iat": int(time.time()),
                         "exp": int(time.time()) + 60}, None, algorithm="none")
    assert client.get("/tickets", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_token_without_exp_is_rejected(client):
    token = jwt.encode({"sub": "sara.ahmed", "role": "employee", "iat": int(time.time())},
                       auth.JWT_SECRET, algorithm="HS256")
    assert client.get("/tickets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_expired_token_is_rejected(client):
    token = jwt.encode({"sub": "sara.ahmed", "role": "employee", "iat": int(time.time()) - 100,
                        "exp": int(time.time()) - 10}, auth.JWT_SECRET, algorithm="HS256")
    assert client.get("/tickets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_of_deleted_user_is_rejected(client):
    token = auth.create_access_token({"username": "ghost", "role": "employee"})
    assert client.get("/tickets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_role_must_match_current_user_record(client):
    # توكن صادر كموظف لمستخدم دوره الحالي عميل (مثلًا بعد خفض صلاحياته) → مرفوض.
    token = auth.create_access_token({"username": "cust", "role": "employee"})
    assert client.get("/tickets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_customer_token_cannot_reach_employee_endpoints(client):
    assert client.get("/tickets", headers=bearer("cust")).status_code == 403
    assert client.post("/draft", json={"customer_message": "x"}, headers=bearer("cust")).status_code == 403


def test_login_success_and_token_works(client):
    resp = client.post("/login", json={"username": "sara.ahmed", "password": PASSWORD})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert client.get("/tickets", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_login_wrong_password(client):
    assert client.post("/login", json={"username": "sara.ahmed", "password": "nope"}).status_code == 401


def test_login_lockout_after_repeated_failures(client):
    for _ in range(auth.MAX_FAILED_LOGINS):
        assert client.post("/login", json={"username": "sara.ahmed", "password": "nope"}).status_code == 401
    # حتى كلمة المرور الصحيحة تُرفض أثناء القفل.
    assert client.post("/login", json={"username": "sara.ahmed", "password": PASSWORD}).status_code == 429


def test_api_key_auth(client, monkeypatch):
    monkeypatch.setattr(service, "EMPLOYEE_KEYS", {"emp-key-123"})
    monkeypatch.setattr(service, "CUSTOMER_KEYS", {"cust-key-456"})
    assert client.get("/tickets", headers={"X-API-Key": "emp-key-123"}).status_code == 200
    assert client.get("/tickets", headers={"X-API-Key": "cust-key-456"}).status_code == 403
    assert client.get("/tickets", headers={"X-API-Key": "wrong"}).status_code == 401


def test_submit_ticket_is_public_and_validated(client):
    ok = client.post("/submit-ticket", json={"customer_message": "حولت فلوس ومش واصلة", "customer_email": "a@b.co"})
    assert ok.status_code == 200 and ok.json()["status"] == "pending"
    assert client.post("/submit-ticket", json={"customer_message": "x", "customer_email": "not-an-email"}).status_code == 422
    assert client.post("/submit-ticket", json={"customer_message": ""}).status_code == 422
    assert client.post("/submit-ticket", json={"customer_message": "x" * (service.MAX_MESSAGE_CHARS + 1)}).status_code == 422


def test_ticket_ids_are_not_predictable(client):
    ids = {client.post("/submit-ticket", json={"customer_message": "same"}).json()["ticket_id"] for _ in range(5)}
    assert len(ids) == 5


def test_submit_ticket_rejects_html_in_email(client):
    # الإيميل يُعرض في لوحة الموظف (حيث يُخزَّن توكن الجلسة) — لا وسوم HTML.
    payload = {"customer_message": "x", "customer_email": "<img/src=x/onerror=alert(1)>@a.bc"}
    assert client.post("/submit-ticket", json=payload).status_code == 422


def test_users_from_secret_when_file_missing(client, monkeypatch, tmp_path):
    """على السيرفر users.json غير مرفوع: الموظفون من الإعداد السري SANAD_USERS_JSON."""
    import json
    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("SANAD_USERS_JSON", json.dumps([{"username": "ops", "display_name": "Ops", "role": "employee",
                                                        "password_hash": auth.hash_password(PASSWORD)}]))
    assert client.post("/login", json={"username": "ops", "password": PASSWORD}).status_code == 200


def test_root_redirects_to_app(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307) and resp.headers["location"] == "/app/"
