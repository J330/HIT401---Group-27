FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# System dependency needed by PyTorch / ONNX Runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch torchvision \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the Django project
COPY . /app

# Collect CSS / JS
RUN python manage.py collectstatic --noinput

# Download rembg background-removal model during build
RUN python -c "from rembg import new_session; new_session('u2netp')" \
    || echo "u2netp download skipped; rembg will retry when the app runs"

# Start Django through Gunicorn
CMD ["sh", "-c", "exec gunicorn website.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 0 --access-logfile - --error-logfile -"]