# Sanad — صورة Docker للنشر على Hugging Face Spaces (أو أي منصة Docker).
# نمط Spaces الموصى به: مستخدم غير root بـ uid 1000، والتطبيق تحت $HOME/app.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 1000 user

# المكتبات (كـ root). torch نسخة الـ CPU فقط: حوالي 200MB بدل عدة GB لنسخة CUDA.
COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SANAD_EMBED_DEVICE=cpu

# تحميل نموذج الـ embedding داخل الصورة وقت البناء: السيرفر لا يحمّل 1.2GB عند كل تشغيل.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"

WORKDIR $HOME/app
COPY --chown=user . .
RUN mkdir -p data stage2_hybrid/store logs

# Spaces يوجّه الطلبات للمنفذ 7860 (app_port في README.md).
EXPOSE 7860
CMD ["uvicorn", "stage4_production.service:app", "--host", "0.0.0.0", "--port", "7860"]
