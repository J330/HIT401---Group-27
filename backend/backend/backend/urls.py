"""Single-server routes for the frontend and prediction API."""
from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html"), name="home"),
    path("index.html", TemplateView.as_view(template_name="index.html"), name="index"),
    path("upload.html", TemplateView.as_view(template_name="upload.html"), name="upload"),
    path("result.html", TemplateView.as_view(template_name="result.html"), name="result"),
    path("about.html", TemplateView.as_view(template_name="about.html"), name="about"),
    path("admin/", admin.site.urls),
    path("api/", include("detector.urls")),
]
