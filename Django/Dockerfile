FROM python:3.11-slim

# Keep Python output visible in Cloud Run logs and avoid .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# onnxruntime/rembg and PyTorch can require libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the Django application, including trainedai/ when you add it beside manage.py.
COPY . /app

# Collect CSS/JS for WhiteNoise while the image is being built.
RUN python manage.py collectstatic --noinput

# Pre-download the small U2Net background-removal model into the image so a
# new Cloud Run instance does not have to download it on the first scan.
RUN python -c "from rembg import new_session; new_session('u2netp')" \
    || echo "u2netp pre-download skipped; rembg will retry at runtime"

# Cloud Run injects PORT. One worker avoids loading the large DINOv2 model
# multiple times into memory. Threads still allow the web server to respond.
CMD ["sh", "-c", "exec gunicorn website.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 0 --access-logfile - --error-logfile -"]
