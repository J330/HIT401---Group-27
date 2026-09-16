# backend/detector/serializers.py

from rest_framework import serializers
from .models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = [
            "id", "image", "label", "confidence", "is_known_disease",
            "is_banana_leaf", "heatmap", "created_at",
        ]
