"""
المرحلة 4: خدمة الإنتاج (Sanad API)
---------------------------------------------
الفرق الجوهري عن نظام عملاء نهائي: هنا موظف دعم بشري يراجع كل مسودة قبل
إرسالها دائمًا، فالقرار المهم ليس "أجب أم ارفض" (ثنائي)، بل تصنيف ثلاثي لمدى
الثقة في المسودة:

  - send_ready:   ثقة عالية + مصدر KB رسمي → الموظف يراجع بسرعة ويرسل.
  - needs_review: ثقة متوسطة، أو المصدر تذكرة سابقة (سابقة غير رسمية) → الموظف
                  يقرأ بعناية ويتحقق من السياسة الرسمية قبل الإرسال.
  - escalate:     ثقة منخفضة جدًا أو لا مصدر مناسب → لا نعرض مسودة قد تُضلّل،
                  نطلب تصعيد الحالة مباشرة (نفس الفجوة التي كشفها تقييم المرحلة 3
                  لسؤال "do you sell iPhones").

الثقة احتمال معايَر (راجع stage2_hybrid/confidence.py) وليست نسبة كلمات مشتركة.

الاسترجاع يتم دائمًا من كل المصادر المتاحة (كل مقالات KB وكل التذاكر السابقة)
بدون تقييد بالأقسام. تسجيل لكل طلب، وCache بسيط.

شغّله من داخل مجلد المشروع (sanad):
    uvicorn stage4_production.service:app --reload
ثم (بعد POST /login للحصول على توكن):
    curl -X POST http://127.0.0.1:8000/draft -H "Content-Type: application/json" \\
         -H "Authorization: Bearer <token>" \\
         -d '{"customer_message": "حسابي اتجمد وأنا مش عارف ليه"}'
"""

import hashlib

import common
import hmac
import logging
import os
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common import generate_draft
from stage2_hybrid import rag
from stage2_hybrid.rag import retrieve_with_signals, build_index, FINAL_K
from stage2_hybrid import confidence as confidence_model
from stage2_hybrid.confidence import ESCALATE_BELOW, SEND_READY_ABOVE
from stage4_production import auth, ticket_store
from stage4_production.email_service import send_reply_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sanad")

MAX_MESSAGE_CHARS = 4000

# العنوان العام الذي يفتح منه العميل رابط الرد (run.bat يضبطه على المنفذ المختار).
# لعملاء حقيقيين يجب أن يكون عنوانًا عامًا لا localhost.
PUBLIC_URL = (os.environ.get("SANAD_PUBLIC_URL") or "http://localhost:8000").rstrip("/")

_draft_cache: dict[str, dict] = {}
_llm_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


# ---------------------------------------------------------------------------
# صلاحيات: موظف مقابل عميل
# ---------------------------------------------------------------------------
# القرار التصميمي المتعمد: العميل لا يحصل أبدًا على وصول مباشر لمسودات RAG عبر
# /draft، حتى لو كانت الثقة عالية جدًا — هذا يتناقض مع المبدأ الأساسي للمشروع
# بالكامل ("الموظف البشري يراجع دائمًا قبل الإرسال"، راجع رأس هذا الملف). لو
# سمحنا للعميل بالوصول المباشر لـ /draft، يتحول النظام فعليًا لبوت عملاء نهائي
# بمخاطر Hallucination مباشرة على العميل، وهو تحديدًا ما صُمم هذا المشروع
# لتجنبه من البداية. لذلك: العميل له نقطة دخول واحدة فقط (/submit-ticket)
# تُسجِّل رسالته لمراجعة موظف لاحقًا عبر /draft، ولا شيء غير ذلك.
#
# مصادقتان مدعومتان معًا:
#   1) تسجيل دخول حقيقي (POST /login -> JWT عبر ترويسة Authorization: Bearer)
#      — المسار الأساسي لواجهة الموظف الفعلية (راجع auth.py).
#   2) مفتاح X-API-Key ثابت (متغيرات بيئة) — يبقى مدعومًا للتوافق مع
#      الاختبارات والأتمتة البسيطة، وكـ مسار بديل بلا تسجيل دخول تفاعلي.
#
# مغلق افتراضيًا (fail-closed): طلب بلا توكن صالح ولا مفتاح صالح يُرفض دائمًا
# بـ 401. سابقًا كان غياب مفاتيح X-API-Key في .env يجعل *أي* طلب مجهول يُعامَل
# كموظف (تجاوز كامل للمصادقة رغم وجود مستخدمين حقيقيين في users.json). الوضع
# المفتوح للتطوير المحلي صار يتطلب تفعيلًا صريحًا: SANAD_DEV_OPEN=1.
def _parse_keys(env_name: str) -> set[str]:
    raw = os.environ.get(env_name, "")
    return {k.strip() for k in raw.split(",") if k.strip()}


EMPLOYEE_KEYS = _parse_keys("SANAD_EMPLOYEE_KEYS")
CUSTOMER_KEYS = _parse_keys("SANAD_CUSTOMER_KEYS")
DEV_OPEN = os.environ.get("SANAD_DEV_OPEN", "").lower() in ("1", "true", "yes")
if DEV_OPEN:
    logger.warning("SANAD_DEV_OPEN مفعّل: الطلبات بلا مصادقة تُعامَل كموظف. لا تستخدمه خارج جهازك.")

DEV_IDENTITY = {"sub": "dev", "role": "employee", "display_name": "موظف (وضع تطوير)"}


def _key_in(candidate: str, keys: set[str]) -> bool:
    # مقارنة ثابتة الزمن (لا تسريب لمحتوى المفتاح عبر توقيت المقارنة).
    return any(hmac.compare_digest(candidate.encode(), k.encode()) for k in keys)


def get_identity(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """يحدد هوية الطالب. الترتيب: توكن JWT (من /login) ثم X-API-Key ثم وضع
    التطوير المفتوح (فقط إن فُعِّل صراحة). غير ذلك → 401."""
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="ترويسة Authorization غير صالحة")
        payload = auth.decode_access_token(authorization.removeprefix("Bearer ").strip())
        if payload is None:
            raise HTTPException(status_code=401, detail="جلسة الدخول منتهية أو غير صالحة، سجّل الدخول مرة أخرى")
        return payload

    if x_api_key:
        if _key_in(x_api_key, EMPLOYEE_KEYS):
            return {"sub": "api-key", "role": "employee", "display_name": "مفتاح API (موظف)"}
        if _key_in(x_api_key, CUSTOMER_KEYS):
            return {"sub": "api-key", "role": "customer", "display_name": "مفتاح API (عميل)"}
        raise HTTPException(status_code=401, detail="مفتاح X-API-Key غير صالح")

    if DEV_OPEN:
        return DEV_IDENTITY
    raise HTTPException(status_code=401, detail="لازم تسجّل الدخول (Authorization: Bearer) أو ترويسة X-API-Key صالحة")


def get_current_employee(identity: dict = Depends(get_identity)) -> dict:
    if identity.get("role") != "employee":
        raise HTTPException(status_code=403, detail="هذا الطلب متاح لموظفي الدعم فقط")
    return identity


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    build_index()
    t1 = time.perf_counter()
    confidence_model.reset_model()
    confidence_model.get_model()
    t2 = time.perf_counter()
    logger.info(f"STARTUP | embedding model + index={(t1 - t0) * 1000:.0f}ms | confidence fit={(t2 - t1) * 1000:.0f}ms")
    logger.info("الفهرس ونموذج الثقة جاهزان، الخدمة بدأت.")
    yield


app = FastAPI(title="Sanad", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    display_name: str


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    if auth.is_locked_out(req.username):
        raise HTTPException(status_code=429, detail="محاولات دخول فاشلة كثيرة، حاول بعد قليل")
    user = auth.authenticate(req.username, req.password)
    if user is None:
        auth.record_failed_login(req.username)
        # رسالة عامة عمدًا (لا نوضح هل اسم المستخدم خطأ أم كلمة المرور) — تقليل
        # تسريب المعلومات لمن يحاول تخمين حسابات صالحة.
        raise HTTPException(status_code=401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")
    auth.clear_failed_logins(req.username)
    token = auth.create_access_token(user)
    logger.info(f"LOGIN | username={user['username']}")
    return LoginResponse(access_token=token, role=user["role"], display_name=user.get("display_name", user["username"]))


class DraftRequest(BaseModel):
    # ticket_id (الواجهة): تُبنى الرسالة من محادثة التذكرة كلها (الرسالة الأولى +
    # توضيحات العميل). customer_message: رسالة مباشرة (توافقية/أتمتة).
    customer_message: Optional[str] = Field(default=None, min_length=1, max_length=MAX_MESSAGE_CHARS)
    ticket_id: Optional[str] = Field(default=None, max_length=64)


class Citation(BaseModel):
    title: str
    source_type: str  # "kb" أو "manual" (دليل السياسات) أو "past_ticket"


class DraftResponse(BaseModel):
    draft: str
    citations: list[Citation]
    confidence: float
    action: str  # send_ready | needs_review | clarify | escalate
    cached: bool
    latency_ms: float
    # سبب التصعيد بلغة الموظف (None إن لم تُصعَّد الحالة). عند التصعيد لا تُكتب
    # مسودة، لكن تُعرض المصادر المسترجعة مع السبب ليقرر الموظف بنفسه.
    reason: Optional[str] = None
    # action == "clarify": حتى 3 أسئلة للعميل (والمسودة = إيميل يحتويها).
    clarifying_questions: list[str] = []
    # زمن كل مرحلة بالملّي ثانية (لتشخيص البطء): retrieval، translation، rerank،
    # wait_llm (انتظار الترجمة/الترتيب بعد انتهاء الاسترجاع)، confidence، draft.
    timings: dict[str, float] = {}


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def escalation_reason(confidence: float, has_sources: bool, translated: bool) -> str:
    if not has_sources:
        return "لا توجد مصادر في قاعدة المعرفة أو دليل السياسات أو التذاكر السابقة لهذه الرسالة."
    pct = round(confidence * 100)
    if not translated:
        return (f"الثقة منخفضة ({pct}%) لأن ترجمة رسالة العميل تعذّرت (خدمة الـ LLM لم تستجب)، "
                "والثقة بدون الترجمة تكون أقل عادةً لرسائل العامية. راجع المصادر أدناه — قد تكون مناسبة — "
                "أو أعد توليد المسودة بعد قليل.")
    return (f"الثقة منخفضة ({pct}%): المصادر المسترجعة غالبًا لا تجيب على رسالة العميل. "
            "راجعها أدناه، واكتب الرد يدويًا أو صعّد الحالة.")


def decide_action(confidence: float, top_chunk: Optional[dict]) -> str:
    if top_chunk is None or confidence < ESCALATE_BELOW:
        return "escalate"
    if confidence > SEND_READY_ABOVE and top_chunk["source_type"] == "kb":
        return "send_ready"
    return "needs_review"


def handle_request(customer_message: str, allow_clarify: bool = True) -> DraftResponse:
    start = time.time()
    key = _cache_key(f"{allow_clarify}::{customer_message}")
    if key in _draft_cache:
        cached = _draft_cache[key]
        latency = (time.time() - start) * 1000
        logger.info(f"CACHE HIT | msg={customer_message!r}")
        return DraftResponse(**{**cached, "cached": True, "latency_ms": round(latency, 2)})

    # البحث في كل المصادر المتاحة. ترتيب المرشحين يعتمد على الرسالة الأصلية فقط،
    # فترجمة الـ LLM (المطلوبة لحساب الثقة) وترتيب الـ LLM مستقلان ويعملان بالتوازي
    # بدل طلبين متتاليين لـ OpenRouter.
    timings: dict[str, float] = {}

    def timed(name, fn, *args):
        t = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timings[name] = round((time.perf_counter() - t) * 1000, 1)

    translation_future = _llm_pool.submit(timed, "translation", rag.translate_query_for_retrieval, customer_message)
    candidates, _ = timed("retrieval", retrieve_with_signals, customer_message, 8, False)
    rerank_future = _llm_pool.submit(timed, "rerank", rag.llm_rerank, customer_message, candidates) if candidates else None
    t_wait = time.perf_counter()
    translation = translation_future.result()
    reranked = rerank_future.result() if rerank_future else None
    timings["wait_llm"] = round((time.perf_counter() - t_wait) * 1000, 1)
    candidates, signals = timed("signals", retrieve_with_signals, customer_message, 8, False, translation)
    top = (reranked or candidates[:FINAL_K]) if candidates else []

    # نموذج الثقة مُعايَر على ميزات أول نتيجة في الترتيب الهجين (هامش RRF،
    # اتفاق المؤشرات). ترتيب LLM قد يقدّم مصدرًا آخر فتبدو ميزاته الترتيبية
    # "ضعيفة" ظلمًا، لذا نأخذ أعلى احتمال بين المصادر النهائية المعروضة.
    t = time.perf_counter()
    model = confidence_model.get_model()
    confidence = max((model.predict(customer_message, c, signals) for c in top), default=0.0)
    action = decide_action(confidence, top[0] if top else None)
    timings["confidence"] = round((time.perf_counter() - t) * 1000, 1)

    # القرار لنموذج الثقة المعايَر وحده: تحقق "هل السياق كافٍ؟" بالـ LLM أُزيل بعد
    # أن أخطأ في قراريه في التقييم (evaluation/generation_eval.py) — مرّر سؤالًا
    # خارج النطاق (قروض) وصعّد سؤالًا صحيحًا (غلق الحساب).

    # المصادر تُعرض دائمًا (حتى عند التصعيد) ليقرر الموظف بنفسه.
    citations = [Citation(title=c["title"], source_type=c["source_type"]) for c in top]
    translated = bool(translation)
    questions: list[str] = []
    if action == "escalate" and top and allow_clarify:
        # رسالة غير مفهومة: الـ LLM يقرر في طلب واحد إن كانت خارج النطاق أصلًا
        # (تصعيد كالمعتاد) أم غامضة فيكتب حتى 3 أسئلة يراجعها الموظف ويرسلها.
        clar = timed("clarify", common.generate_clarifying_questions, customer_message, top)
        if clar and not clar["out_of_scope"] and clar["questions"]:
            action, questions = "clarify", clar["questions"]
        elif clar and clar["out_of_scope"]:
            logger.info("  ↳ الـ LLM قدّر أن الرسالة خارج نطاق خدمات المحفظة — تصعيد بدون أسئلة")

    if action == "clarify":
        draft_text = common.compose_clarification_email(questions)
        reason = ("الرسالة غير واضحة بما يكفي لاختيار الرد الصحيح. أرسل للعميل الأسئلة التوضيحية أدناه "
                  "(يمكنك تعديلها)، وسيصله رابط يرد منه على نفس التذكرة.")
    elif action == "escalate":
        draft_text, reason = "", escalation_reason(confidence, bool(top), translated)
    else:
        draft_text, reason = timed("draft", generate_draft, customer_message, top), None

    latency = (time.time() - start) * 1000
    result = {"draft": draft_text, "citations": citations, "confidence": round(confidence, 3),
              "action": action, "reason": reason, "clarifying_questions": questions}
    # نتيجة محسوبة بدون ترجمة بسبب عطل مؤقت لا تُخزَّن: إعادة التوليد بعد قليل يجب
    # أن تجرّب الترجمة من جديد، لا أن ترجع نفس النتيجة من الكاش.
    if translated or not common.get_openrouter_client():
        _draft_cache[key] = result

    logger.info(
        f"msg={customer_message!r} | confidence={confidence:.3f} | "
        f"action={action} | latency_ms={latency:.1f} | timings={timings}"
    )
    return DraftResponse(**result, cached=False, latency_ms=round(latency, 2), timings=timings)


@app.post("/draft", response_model=DraftResponse)
def draft_endpoint(req: DraftRequest, employee: dict = Depends(get_current_employee)):
    if req.ticket_id:
        ticket = ticket_store.get_ticket(req.ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
        # جولة توضيح واحدة لكل تذكرة: بعد رد العميل لا نسأل مرة ثانية.
        return handle_request(ticket_store.conversation_for_retrieval(ticket),
                              allow_clarify=not ticket_store.had_clarification(ticket))
    if not req.customer_message:
        raise HTTPException(status_code=422, detail="أرسل ticket_id أو customer_message")
    return handle_request(req.customer_message)


class TicketSubmission(BaseModel):
    customer_message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    # مطلوب فعليًا لإرسال رد بالإيميل لاحقًا؛ اختياري في الـ schema لسهولة الاختبار
    customer_email: Optional[str] = Field(default=None, max_length=254, pattern=r"^[^@\s<>\"'`]+@[^@\s<>\"'`]+\.[^@\s<>\"'`]+$")


class Message(BaseModel):
    sender: str  # customer | agent
    kind: str    # message | clarification
    text: str
    created_at: float


class TicketRecord(BaseModel):
    ticket_id: str
    customer_message: str          # أول رسالة من العميل
    customer_email: Optional[str] = None
    status: str  # pending | awaiting_customer | sent | escalated
    created_at: float
    messages: list[Message] = []   # المحادثة كاملة


def _record(ticket: dict) -> TicketRecord:
    return TicketRecord(**{k: ticket[k] for k in ("ticket_id", "customer_message", "customer_email",
                                                  "status", "created_at", "messages")})


@app.post("/submit-ticket", response_model=TicketRecord)
def submit_ticket(req: TicketSubmission):
    """نقطة الدخول الوحيدة المتاحة للعميل — عامة عمدًا (صفحة العميل بلا تسجيل
    دخول)، وآمنة لأنها لا تُشغِّل أي استرجاع أو توليد ولا تُرجع أي محتوى من
    قاعدة المعرفة: فقط تسجّل الرسالة بحالة 'pending' لمراجعة موظف لاحقًا."""
    ticket = ticket_store.create_ticket(req.customer_message, req.customer_email)
    logger.info(f"TICKET SUBMITTED | id={ticket['ticket_id']} | msg={req.customer_message!r}")
    return _record(ticket)


@app.get("/tickets", response_model=list[TicketRecord])
def list_tickets(status: str = "pending", employee: dict = Depends(get_current_employee)):
    """موظفون فقط: قائمة التذاكر (افتراضيًا pending) لمراجعتها عبر /draft."""
    return [_record(t) for t in ticket_store.list_tickets(status)]


class ResolveRequest(BaseModel):
    final_text: str = Field(max_length=10000)
    # "send": رد نهائي بالإيميل | "clarify": أسئلة توضيحية بالإيميل + رابط رد على
    # نفس التذكرة (تصبح awaiting_customer) | "escalate": بدون إرسال
    resolution: str


class ResolveResponse(BaseModel):
    ticket: TicketRecord
    email_sent: bool
    email_error: Optional[str] = None
    reply_link: Optional[str] = None  # للأسئلة التوضيحية: ليرسله الموظف يدويًا إن فشل الإيميل


def reply_link(ticket: dict) -> str:
    return f"{PUBLIC_URL}/app/reply.html?t={ticket['reply_token']}"


@app.post("/tickets/{ticket_id}/resolve", response_model=ResolveResponse)
def resolve_ticket(ticket_id: str, req: ResolveRequest, employee: dict = Depends(get_current_employee)):
    """موظفون فقط. 'send' و'clarify' يرسلان final_text فعليًا لبريد العميل عبر
    SendGrid ('clarify' يضيف رابط الرد). فشل الإرسال لا يمنع الموظف من إكمال عمله:
    تبقى التذكرة pending ويُبلَّغ بالخطأ (ومعه الرابط) ليكمل يدويًا. 'escalate' لا
    يرسل شيئًا، يعلّم التذكرة فقط لمتابعة فريق أعلى."""
    ticket = ticket_store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    if req.resolution not in ("send", "clarify", "escalate"):
        raise HTTPException(status_code=422, detail="resolution يجب أن تكون 'send' أو 'clarify' أو 'escalate'")

    email_sent, email_error, link = False, None, None
    if req.resolution in ("send", "clarify"):
        body = req.final_text
        if req.resolution == "clarify":
            link = reply_link(ticket)
            body = f"{body}\n\nللرد على الأسئلة: {link}"
        if not ticket.get("customer_email"):
            email_error = "لا يوجد بريد إلكتروني مسجَّل لهذا العميل — أرسل الرد يدويًا."
        else:
            email_sent, email_error = send_reply_email(
                to_email=ticket["customer_email"],
                subject="أسئلة بخصوص تذكرتك" if req.resolution == "clarify" else "رد بخصوص تذكرتك",
                body_text=body,
            )
        # لا تتغير الحالة إلا لو الإيميل اتبعت فعلًا؛ فشل الإرسال يبقيها "pending"
        # حتى لا تختفي من قائمة المتابعة رغم أن العميل لم يستلم شيئًا.
        if email_sent:
            kind = "clarification" if req.resolution == "clarify" else "message"
            ticket_store.add_message(ticket_id, "agent", req.final_text, kind)
            ticket_store.set_status(ticket_id, "awaiting_customer" if req.resolution == "clarify" else "sent")
    else:
        ticket_store.set_status(ticket_id, "escalated")

    ticket = ticket_store.get_ticket(ticket_id)
    logger.info(
        f"TICKET RESOLVED | id={ticket_id} | by={employee.get('display_name')} | "
        f"resolution={req.resolution} | email_sent={email_sent}" + (f" | error={email_error}" if email_error else "")
    )
    return ResolveResponse(ticket=_record(ticket), email_sent=email_sent, email_error=email_error, reply_link=link)


# ---------------------------------------------------------------------------
# رد العميل على الأسئلة التوضيحية (عام، بالرمز السري في الرابط فقط)
# ---------------------------------------------------------------------------
class ReplyView(BaseModel):
    questions: str        # آخر رسالة أسئلة من الموظف
    can_reply: bool       # False إذا رد العميل بالفعل أو أُغلقت التذكرة


class CustomerReply(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


def _ticket_by_token(token: str) -> dict:
    ticket = ticket_store.get_by_reply_token(token) if len(token) <= 64 else None
    if ticket is None:
        raise HTTPException(status_code=404, detail="الرابط غير صالح أو منتهي")
    return ticket


@app.get("/reply/{token}", response_model=ReplyView)
def reply_view(token: str):
    """لا يكشف أي بيانات غير أسئلة الموظف نفسها (لا إيميل ولا رقم تذكرة)."""
    ticket = _ticket_by_token(token)
    last_q = next((m["text"] for m in reversed(ticket["messages"]) if m["kind"] == "clarification"), "")
    return ReplyView(questions=last_q, can_reply=ticket["status"] == "awaiting_customer")


@app.post("/reply/{token}")
def reply_submit(token: str, req: CustomerReply):
    ticket = _ticket_by_token(token)
    if ticket["status"] != "awaiting_customer":
        raise HTTPException(status_code=409, detail="تم استلام ردك بالفعل، وسيتواصل معك أحد موظفي الدعم.")
    ticket_store.add_message(ticket["ticket_id"], "customer", req.message)
    ticket_store.set_status(ticket["ticket_id"], "pending")   # ترجع لقائمة الموظف
    logger.info(f"CUSTOMER REPLY | id={ticket['ticket_id']} | msg={req.message!r}")
    return {"status": "received"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app/")


@app.get("/health")
def health():
    # instance: رمز عشوائي يمرره run.bat لكل تشغيل، ليتأكد أن من يرد على المنفذ هو
    # هذا السيرفر تحديدًا لا تطبيقًا آخر (مثل نسخة Docker على نفس المنفذ).
    return {"status": "ok", "instance": os.environ.get("SANAD_INSTANCE_ID", "")}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="app")
