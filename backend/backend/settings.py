# backend/backend/settings.py
# Sources: https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/
# https://whitenoise.readthedocs.io/
# https://github.com/adamchainz/django-cors-headers

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Cloud Run puts this Django app behind a HTTPS reverse proxy.
DEBUG = os.environ.get("DEBUG", "True").lower() in {"true", "1", "yes"}
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    if not DEBUG:
        from django.core.exceptions import ImproperlyConfigured
        raise ImproperlyConfigured("Set SECRET_KEY using Google Secret Manager before deploying.")
    SECRET_KEY = "local-development-only-unsafe-key"

# A leading dot matches subdomains such as SERVICE-HASH-REGION.a.run.app.
# Add your custom hostname through ALLOWED_HOSTS when configuring a domain.
ALLOWED_HOSTS = [host.strip() for host in
                 os.environ.get("ALLOWED_HOSTS", ".run.app,localhost,127.0.0.1").split(",")
                 if host.strip()]
CSRF_TRUSTED_ORIGINS = ["https://*.run.app"]
CSRF_TRUSTED_ORIGINS += [origin.strip() for origin in
                        os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")
                        if origin.strip()]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# Cloud Run's filesystem is ephemeral. By default it does not save prediction
# rows, and no Cloud SQL instance is needed just to run public image scans.
# If you want a durable scan history later, add Cloud SQL and enable this.
SAVE_PREDICTIONS = os.environ.get("SAVE_PREDICTIONS", "false" if os.environ.get("K_SERVICE") else "true").lower() in {"true", "1", "yes"}

# Select a model whose corresponding trained .pt file is present in ml/models.
# The Docker image downloads matching pretrained backbones at build time.
# For a different model, change the build argument AND Cloud Run env setting.
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
        "DIRS": [BASE_DIR / "templates"],
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

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
"default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
"staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Same-origin frontend; no cross-origin access is necessary on Cloud Run.
# You can explicitly add trusted frontend origins for separate deployments.
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in
                        os.environ.get("FRONTEND_ORIGIN", "").split(",")
                        if origin.strip()]

# Reject huge uploads before they ever reach a model
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
