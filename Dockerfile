# Build this from the REPOSITORY ROOT, not backend/: docker build -f Dockerfile .
# Python 3.11 is supported by the project's inference dependencies.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    DEBUG=False \
    SAVE_PREDICTIONS=False

ARG ACTIVE_MODEL=convnextv2
ENV ACTIVE_MODEL=${ACTIVE_MODEL}

WORKDIR /app
# Install CPU-only Torch to avoid the huge, unnecessary CUDA wheels.
COPY backend/requirements-cloudrun.txt /tmp/requirements-cloudrun.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r /tmp/requirements-cloudrun.txt

COPY backend/ /app/backend/
COPY cloudrun/ /app/cloudrun/
COPY ml/models/ /app/ml/models/

# Fail the build clearly if the checkpoint files are absent or Git LFS pointers.
RUN python /app/cloudrun/check_models.py --model "${ACTIVE_MODEL}" --root /app
# At runtime Cloud Run can use the cached model base weights without depending
# on Hugging Face downloads during the first scan.
RUN python /app/cloudrun/precache_models.py
ENV HF_HUB_OFFLINE=1

WORKDIR /app/backend
# Collect static assets during the build with a throwaway SECRET_KEY.
RUN SECRET_KEY=static-collection-only python manage.py collectstatic --noinput

# Least privilege; Cloud Run filesystem remains ephemeral.
RUN useradd --uid 10001 --create-home appuser && \
    chown -R appuser:appuser /app /opt/huggingface
USER appuser

# Cloud Run sets PORT itself (usually 8080); listen on all container interfaces.
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn backend.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 900 --access-logfile - --error-logfile -"]
