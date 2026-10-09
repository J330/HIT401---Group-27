# backend/detector/admin.py
# Source: https://docs.djangoproject.com/en/5.0/ref/contrib/admin/

from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "label", "confidence", "is_known_disease", "is_banana_leaf", "created_at")
    list_filter = ("label", "is_known_disease", "is_banana_leaf")
