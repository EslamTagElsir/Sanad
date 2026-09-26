---
title: Sanad
emoji: 🤝
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Sanad · سند

**مساعد موظف الدعم لمحفظة إلكترونية** — يقرأ رسالة العميل (فصحى، عامية مصرية، أو إنجليزي)، يبحث في قاعدة المعرفة ودليل سياسات الشركة والتذاكر السابقة، ويكتب **مسودة رد** يراجعها موظف بشري قبل الإرسال. لا يصل أي شيء للعميل بدون مراجعة الموظف.

An agent-assist copilot for e-wallet customer support: retrieval-augmented reply drafts in Arabic (incl. Egyptian dialect) and English, with calibrated confidence and a human always in the loop.

## المميزات

- **استرجاع دلالي** بنموذج `Qwen/Qwen3-Embedding-0.6B` متعدد اللغات (يعمل محليًا على CPU) من ثلاثة مصادر: مقالات قاعدة المعرفة، دليل السياسات (يُستخرج نصه من ملف PDF)، والتذاكر المحلولة سابقًا.
- **ثقة معايَرة** (Logistic Regression على المجموعة الذهبية) تقرر: `جاهز للإرسال` / `يحتاج مراجعة` / `يحتاج توضيح` / `تصعيد`.
- **أسئلة توضيحية**: إذا كانت الرسالة غامضة يكتب حتى 3 أسئلة للعميل، يراجعها الموظف ويرسلها، ويرد العميل من رابط على نفس التذكرة.
- **عند التصعيد** تظهر المصادر المسترجعة وسبب التصعيد بدل مسودة قد تُضلّل.
- **LLM عبر OpenRouter** لترجمة الرسالة وإعادة ترتيب المصادر وكتابة المسودة (موديل واحد قابل للتغيير).
- **مصادقة JWT** للموظفين، والتذاكر والمحادثات في SQLite، وإرسال الرد بالإيميل عبر SendGrid.

## هيكل المشروع

```
common.py                  تحميل البيانات، تقسيم المستندات، استخراج PDF، استدعاءات OpenRouter، المسودة والأسئلة
stage2_hybrid/             الاسترجاع: embeddings.py، rag.py (الفهرس والبحث)، confidence.py (الثقة المعايَرة)
stage4_production/         الخدمة (FastAPI): service.py، auth.py، ticket_store.py، email_service.py، static/
data/                      قاعدة المعرفة، التذاكر السابقة، المجموعة الذهبية وترجماتها
docs/company_policies.pdf  دليل سياسات الشركة وإجراءات الدعم (مصدر استرجاع)
evaluation/                التقييم (golden_eval، generation_eval، benchmark_latency) والاختبارات
```

## التشغيل محليًا

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # ضع مفتاح OpenRouter و SANAD_JWT_SECRET
python -m stage4_production.manage_users add sara.ahmed --name "سارة أحمد"
uvicorn stage4_production.service:app --port 8000
```

ثم افتح `http://localhost:8000/app/`. على Windows مع WSL: `run.bat` يفعل كل ذلك ويختار منفذًا فارغًا تلقائيًا.

## الاختبارات والتقييم

```bash
python -m pytest evaluation/tests          # اختبارات (بدون استدعاءات LLM)
python -m evaluation.golden_eval           # جودة الاسترجاع ومعايرة الثقة
python -m evaluation.generation_eval       # جودة المسودات (يستهلك طلبات OpenRouter)
```

## النشر على Hugging Face Spaces

1. أنشئ Space جديدًا من نوع **Docker** (Blank) باسم `sanad`.
2. في **Settings → Variables and secrets** أضف:

   | الاسم | النوع | القيمة |
   |---|---|---|
   | `OPENROUTER_API_KEY` | Secret | مفتاح OpenRouter |
   | `SANAD_JWT_SECRET` | Secret | نص عشوائي طويل (48 حرفًا أو أكثر) |
   | `SANAD_USERS_JSON` | Secret | ناتج `python -m stage4_production.manage_users add ...` |
   | `SANAD_PUBLIC_URL` | Variable | `https://<username>-sanad.hf.space` (لروابط رد العميل) |
   | `OPENROUTER_MODELS` | Variable | اختياري — الافتراضي `inclusionai/ling-3.0-flash-fin:free` |
   | `SENDGRID_API_KEY` / `SENDGRID_FROM_EMAIL` | Secret | اختياري — لإرسال الردود بالإيميل |
   | `SANAD_DB_PATH` | Variable | `/data/tickets.db` إذا فعّلت التخزين الدائم |

3. ادفع الكود إلى الـ Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<username>/sanad
   git push space main
   ```

   البناء الأول يأخذ عدة دقائق (تثبيت PyTorch وتحميل نموذج الـ embedding داخل الصورة)، وبدء التشغيل حوالي دقيقة.

**تنبيهات:**
- بدون **التخزين الدائم** (Persistent storage، مدفوع) تضيع التذاكر والمحادثات مع كل إعادة تشغيل أو نشر.
- المساحات المجانية تنام بعد فترة عدم استخدام، وأول طلب بعدها يأخذ حوالي دقيقة.
- لا يصلح النشر على Vercel: حجم المكتبات والنموذج (>2GB) وزمن البدء والحاجة لتخزين دائم لا تناسب الدوال اللحظية (serverless).

## النشر على Cloudflare (Containers + D1)

نفس الـ Dockerfile يعمل داخل Cloudflare Container خلف Worker صغير (`cloudflare/`).
**تذاكر المتابعة فقط** (التي أُرسلت فيها أسئلة توضيحية) تُحفظ في قاعدة D1 وتُحذف
عند إغلاقها؛ التذاكر الجديدة على قرص الحاوية وتضيع إذا نامت الحاوية (بعد 30 دقيقة
خمول) قبل أن يفتحها موظف.

المتطلبات: Node.js، Docker، خطة Workers Paid (5$ شهريًا). ثم من مجلد `cloudflare/`:

```bash
npm install
npx wrangler login
npx wrangler d1 create sanad-followups          # ضع database_id الناتج في wrangler.jsonc
npx wrangler d1 migrations apply sanad-followups --remote
npx wrangler secret put OPENROUTER_API_KEY
npx wrangler secret put SANAD_JWT_SECRET
npx wrangler secret put SANAD_USERS_JSON        # ناتج manage_users add
npx wrangler deploy                             # يبني الصورة (~2.5GB) ويرفعها
```

بعد أول نشر عدّل `SANAD_PUBLIC_URL` في `wrangler.jsonc` لعنوان الـ Worker الفعلي وأعد `wrangler deploy`.

## الترخيص

المشروع مرخّص بـ **GNU AGPL-3.0** (راجع [LICENSE](LICENSE))، لأنه يستخدم PyMuPDF (رخصة AGPL-3.0) لاستخراج نص دليل السياسات. من يشغّل نسخة معدّلة كخدمة عبر الشبكة يجب أن يتيح كودها المصدري لمستخدميها.

Licensed under the GNU AGPL-3.0 — see [LICENSE](LICENSE).
