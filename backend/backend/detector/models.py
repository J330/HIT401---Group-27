# backend/detector/models.py

from django.db import models


class Prediction(models.Model):
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    is_known_disease = models.BooleanField(default=True)
    is_banana_leaf = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)