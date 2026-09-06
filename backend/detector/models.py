# backend/detector/models.py
from django.db import models


class Prediction(models.Model):
    image = models.ImageField(upload_to="uploads/")
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    is_known_disease = models.BooleanField(default=True)
    is_banana_leaf = models.BooleanField(default=True)
    heatmap = models.ImageField(upload_to="heatmaps/", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
