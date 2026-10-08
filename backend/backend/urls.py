# backend/backend/urls.py
# Source: https://docs.djangoproject.com/en/5.0/topics/http/urls/

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("detector.urls")),
]
