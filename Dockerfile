# Based on the simpler, working-style Dockerfile in Django(4).zip.
# Use this Dockerfile from the repository root so backend/ AND ml/models/ are visible.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    DEBUG=False \
    SAVE_PREDICTIONS=False

ARG ACTIVE_MODEL=convnextv2
ENV ACTIVE_MODEL=${ACTIVE_MODEL}

WORKDIR /app
# From the older working Dockerfile: libgomp1 is needed by native ML libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-cloudrun.txt /tmp/requirements-cloudrun.txt
# Install CPU-only PyTorch; CUDA wheels are unnecessary on standard Cloud Run.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements-cloudrun.txt

COPY backend/ /app/backend/
COPY cloudrun/ /app/cloudrun/
COPY ml/models/ /app/ml/models/

# NOTE: Unlike the old strict Dockerfile, the image can build without model
# weights. This allows the *website* to start (like Django(4).zip), but scans
# still require REAL .pt weights. The PowerShell helper / CI workflow validates
# them before deploying a scanning-enabled service.
# Pre-cache the public pretrained backbones so Cloud Run need not fetch them on
# the first scan. This does NOT replace the locally trained .pt weights.
RUN python /app/cloudrun/precache_models.py
ENV HF_HUB_OFFLINE=1

WORKDIR /app/backend
# From Django(4).zip: collect static CSS/JS at image-build time for WhiteNoise.
RUN SECRET_KEY=collectstatic-build-only python manage.py collectstatic --noinput

# Least privilege: one Gunicorn worker avoids duplicating the large models.
RUN useradd --uid 10001 --create-home appuser \
    && chown -R appuser:appuser /app /opt/huggingface
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn backend.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 900 --access-logfile - --error-logfile -"]
