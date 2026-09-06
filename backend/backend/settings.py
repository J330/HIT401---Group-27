# backend/backend/settings.py
# Sources: https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/
#          https://whitenoise.readthedocs.io/
#          https://github.com/adamchainz/django-cors-headers
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = os.environ.get("DEBUG", "True") == "True"
ALLOWED_HOSTS = [os.environ.get("RENDER_EXTERNAL_HOSTNAME", "127.0.0.1"), "localhost"]

# --- Which trained model serves website predictions ------------------------
# One of: 'efficientnetv2s', 'convnextv2', 'swin', 'vit' (the winner from
# ml/evaluate.py). Change the ACTIVE_MODEL environment variable in the
# Render dashboard (or this default for local dev) and restart the server
# to switch models -- no code changes anywhere.
ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "convnextv2")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",  # Django REST Framework -- https://www.django-rest-framework.org/
    "corsheaders",  # django-cors-headers -- https://github.com/adamchainz/django-cors-headers
    "detector",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"

# SQLite for development; swap for Postgres in production via DATABASE_URL
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
"default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
"staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"  # uploaded photos + generated heatmaps land here

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Only the deployed frontend origin may call this API
CORS_ALLOWED_ORIGINS = [os.environ.get("FRONTEND_ORIGIN", "http://127.0.0.1:8080")]

# Reject huge uploads before they ever reach a model
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
