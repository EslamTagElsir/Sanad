"""
المرحلة 4: خدمة الإنتاج (Agent-Assist API)
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

شغّله من داخل مجلد agent-assist-copilot:
    uvicorn stage4_production.service:app --reload
ثم (بعد POST /login للحصول على توكن):
    curl -X POST http://127.0.0.1:8000/draft -H "Content-Type: application/json" \\
         -H "Authorization: Bearer <token>" \\
         -d '{"customer_message": "حسابي اتجمد وأنا مش عارف ليه"}'
"""

import hashlib
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common import generate_draft, openrouter_chat
from stage2_hybrid import rag
from stage2_hybrid.rag import retrieve_with_signals, build_index, FINAL_K
from stage2_hybrid import confidence as confidence_model
from stage2_hybrid.confidence import ESCALATE_BELOW, SEND_READY_ABOVE
from stage4_production import auth
from stage4_production.email_service import send_reply_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("agent_copilot")

MAX_MESSAGE_CHARS = 4000

_draft_cache: dict[str, dict] = {}
_llm_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")
_tickets: dict[str, dict] = {}


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
# المفتوح للتطوير المحلي صار يتطلب تفعيلًا صريحًا: AGENT_ASSIST_DEV_OPEN=1.
def _parse_keys(env_name: str) -> set[str]:
    raw = os.environ.get(env_name, "")
    return {k.strip() for k in raw.split(",") if k.strip()}


EMPLOYEE_KEYS = _parse_keys("AGENT_ASSIST_EMPLOYEE_KEYS")
CUSTOMER_KEYS = _parse_keys("AGENT_ASSIST_CUSTOMER_KEYS")
DEV_OPEN = os.environ.get("AGENT_ASSIST_DEV_OPEN", "").lower() in ("1", "true", "yes")
if DEV_OPEN:
    logger.warning("AGENT_ASSIST_DEV_OPEN مفعّل: الطلبات بلا مصادقة تُعامَل كموظف. لا تستخدمه خارج جهازك.")

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


def _parse_yes_no(text: Optional[str]) -> Optional[bool]:
    """نعم/yes → True، لا/no → False، وأي شيء آخر (رد فارغ من نموذج تفكير
    نفد منه max_tokens، أو كلام غير واضح) → None: لا نبني قرارًا على رد غامض."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t.startswith(("نعم", "yes", "ايوه", "أيوه")):
        return True
    if t.startswith(("لا", "no")):
        return False
    return None


def _llm_scope_check(question: str, chunks: list[dict]) -> Optional[bool]:
    """تحقق ثانٍ عبر OpenRouter للحالات الحدّية فقط (توفيرًا للتكلفة) على كل
    المصادر التي ستُبنى عليها المسودة (لا أولها فقط). رد فارغ أو غامض أو فشل →
    None، فيبقى قرار نموذج الثقة المعايَر كما هو."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    prompt = (
        f"السياق:\n{context}\n\nرسالة العميل: {question}\n\n"
        "هل هذا السياق يحتوي فعلًا على ما يلزم للرد على رسالة العميل هذه؟ أجب بكلمة واحدة: نعم أو لا."
    )
    return _parse_yes_no(openrouter_chat([{"role": "user", "content": prompt}], max_tokens=10, task="scope_check"))


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


app = FastAPI(title="Agent-Assist Copilot - Stage 4", lifespan=lifespan)


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
    customer_message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class Citation(BaseModel):
    title: str
    source_type: str  # "kb" أو "manual" (دليل السياسات) أو "past_ticket"


class DraftResponse(BaseModel):
    draft: str
    citations: list[Citation]
    confidence: float
    action: str  # send_ready | needs_review | escalate
    cached: bool
    latency_ms: float
    # زمن كل مرحلة بالملّي ثانية (لتشخيص البطء): retrieval، translation، rerank،
    # wait_llm (انتظار الترجمة/الترتيب بعد انتهاء الاسترجاع)، confidence،
    # scope_check، draft.
    timings: dict[str, float] = {}


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


ESCALATE_TEXT = "لا توجد معلومات كافية في قاعدة المعرفة أو التذاكر السابقة لصياغة رد موثوق. يُنصح بتصعيد الحالة لموظف أعلى أو فريق مختص."


def decide_action(confidence: float, top_chunk: Optional[dict]) -> str:
    if top_chunk is None or confidence < ESCALATE_BELOW:
        return "escalate"
    if confidence > SEND_READY_ABOVE and top_chunk["source_type"] == "kb":
        return "send_ready"
    return "needs_review"


def handle_request(customer_message: str) -> DraftResponse:
    start = time.time()
    key = _cache_key(customer_message)
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

    # نطاق حدّي حول عتبة التصعيد فقط (حيث الخطأ أخطر: عرض مسودة رغم عدم وجود
    # مصدر مناسب فعلًا). لا نستشير LLM حول عتبة send_ready لأن needs_review هو
    # افتراضي آمن أصلًا هناك (الموظف يراجع في الحالتين).
    if top and abs(confidence - ESCALATE_BELOW) <= 0.1:
        llm_ok = timed("scope_check", _llm_scope_check, customer_message, top)
        if llm_ok is not None:
            action = "needs_review" if llm_ok else "escalate"
            logger.info(f"  ↳ نطاق حدّي (confidence={confidence:.3f})، تحقق LLM: in_scope={llm_ok}")

    if action == "escalate":
        draft_text = ESCALATE_TEXT
        citations = []
    else:
        draft_text = timed("draft", generate_draft, customer_message, top)
        citations = [Citation(title=c["title"], source_type=c["source_type"]) for c in top]

    latency = (time.time() - start) * 1000
    result = {"draft": draft_text, "citations": citations, "confidence": round(confidence, 3), "action": action}
    _draft_cache[key] = result

    logger.info(
        f"msg={customer_message!r} | confidence={confidence:.3f} | "
        f"action={action} | latency_ms={latency:.1f} | timings={timings}"
    )
    return DraftResponse(**result, cached=False, latency_ms=round(latency, 2), timings=timings)


@app.post("/draft", response_model=DraftResponse)
def draft_endpoint(req: DraftRequest, employee: dict = Depends(get_current_employee)):
    return handle_request(req.customer_message)


class TicketSubmission(BaseModel):
    customer_message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    # مطلوب فعليًا لإرسال رد بالإيميل لاحقًا؛ اختياري في الـ schema لسهولة الاختبار
    customer_email: Optional[str] = Field(default=None, max_length=254, pattern=r"^[^@\s<>\"'`]+@[^@\s<>\"'`]+\.[^@\s<>\"'`]+$")


class TicketRecord(BaseModel):
    ticket_id: str
    customer_message: str
    customer_email: Optional[str] = None
    status: str  # pending | sent | escalated
    created_at: float


@app.post("/submit-ticket", response_model=TicketRecord)
def submit_ticket(req: TicketSubmission):
    """نقطة الدخول الوحيدة المتاحة للعميل — عامة عمدًا (صفحة العميل بلا تسجيل
    دخول)، وآمنة لأنها لا تُشغِّل أي استرجاع أو توليد ولا تُرجع أي محتوى من
    قاعدة المعرفة: فقط تسجّل الرسالة بحالة 'pending' لمراجعة موظف لاحقًا."""
    ticket_id = secrets.token_hex(6)
    record = {
        "ticket_id": ticket_id,
        "customer_message": req.customer_message,
        "customer_email": req.customer_email,
        "status": "pending",
        "created_at": time.time(),
    }
    _tickets[ticket_id] = record
    logger.info(f"TICKET SUBMITTED | id={ticket_id} | msg={req.customer_message!r}")
    return TicketRecord(**record)


@app.get("/tickets", response_model=list[TicketRecord])
def list_tickets(status: str = "pending", employee: dict = Depends(get_current_employee)):
    """موظفون فقط: قائمة التذاكر (افتراضيًا pending) لمراجعتها عبر /draft."""
    records = _tickets.values() if status == "all" else (t for t in _tickets.values() if t["status"] == status)
    return [TicketRecord(**t) for t in records]


class ResolveRequest(BaseModel):
    final_text: str = Field(max_length=10000)
    resolution: str  # "send" (إرسال إيميل فعلي للعميل) أو "escalate" (بدون إرسال)


class ResolveResponse(BaseModel):
    ticket: TicketRecord
    email_sent: bool
    email_error: Optional[str] = None


@app.post("/tickets/{ticket_id}/resolve", response_model=ResolveResponse)
def resolve_ticket(ticket_id: str, req: ResolveRequest, employee: dict = Depends(get_current_employee)):
    """موظفون فقط. 'send' يرسل final_text فعليًا لبريد العميل عبر SendGrid
    (raise_for_status لا يُستخدم عمدًا: فشل الإرسال لا يجب أن يمنع الموظف من
    إكمال عمله، فقط يُبلَّغ به ليكمل الإرسال يدويًا). 'escalate' لا يرسل شيئًا،
    يعلّم التذكرة فقط لمتابعة فريق أعلى."""
    ticket = _tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="التذكرة غير موجودة")
    if req.resolution not in ("send", "escalate"):
        raise HTTPException(status_code=422, detail="resolution يجب أن تكون 'send' أو 'escalate'")

    email_sent, email_error = False, None
    if req.resolution == "send":
        if not ticket.get("customer_email"):
            email_error = "لا يوجد بريد إلكتروني مسجَّل لهذا العميل — أرسل المسودة يدويًا."
        else:
            email_sent, email_error = send_reply_email(
                to_email=ticket["customer_email"],
                subject="رد بخصوص تذكرتك",
                body_text=req.final_text,
            )
        # لا نعلّم التذكرة "sent" إلا لو الإيميل اتبعت فعلًا؛ فشل الإرسال يبقيها
        # "pending" حتى لا تختفي من قائمة المتابعة رغم أن العميل لم يستلم شيئًا.
        ticket["status"] = "sent" if email_sent else "pending"
    else:
        ticket["status"] = "escalated"

    logger.info(
        f"TICKET RESOLVED | id={ticket_id} | by={employee.get('display_name')} | "
        f"resolution={req.resolution} | email_sent={email_sent}" + (f" | error={email_error}" if email_error else "")
    )
    return ResolveResponse(ticket=TicketRecord(**ticket), email_sent=email_sent, email_error=email_error)


@app.get("/health")
def health():
    return {"status": "ok"}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="app")
