# backend/detector/views.py
# Source: https://www.django-rest-framework.org/api-guide/views/

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer
from django.views.generic import TemplateView
from django.conf import settings
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class UploadPageView(TemplateView):
    template_name = "upload.html"


class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if image_file is None:
            return Response({"error": "Select an image first."}, status=400)
        # Load heavy AI dependencies only when a scan is requested.
        try:
            from .inference import predict_image
            result = predict_image(image_file)  # Read in memory; stateless on Cloud Run.
        except (OSError, RuntimeError) as exc:
            logger.exception("Prediction model unavailable or invalid")
            if settings.DEBUG:
                detail = str(exc)
            else:
                detail = "The AI models are not available on this deployment. Check the Cloud Run logs and install real trained checkpoints."
            return Response({"error": detail}, status=503)

        # Cloud Run is stateless; avoid writing to ephemeral SQLite there.
        # The report still receives all prediction data without a database ID.
        response_data = {
            "id": None,
            "label": result["label"],
            "confidence": result["confidence"],
            "is_known_disease": result["is_known_disease"],
            "is_banana_leaf": result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
            "created_at": timezone.now().isoformat(),
        }
        if settings.SAVE_PREDICTIONS:
            prediction = Prediction(
                label=result["label"],
                confidence=result["confidence"],
                is_known_disease=result["is_known_disease"],
                is_banana_leaf=result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
            )
            prediction.save()
            response_data = PredictionSerializer(prediction).data

        # Keep grayscale heatmap for the existing interactive viewer, while
        # also exposing the blended overlay from the earlier Django project.
        heatmap_url = None
        overlay_url = None
        heatmap_error = result.get("heatmap_error")
        if result["heatmap_array"] is not None:
            try:
                from PIL import Image
                from .heatmap import make_heatmap_urls
                image_file.seek(0)
                with Image.open(image_file) as image:
                    heatmap_url, overlay_url = make_heatmap_urls(image, result["heatmap_array"])
            except Exception as exc:
                logger.exception("Failed to encode a Grad-CAM heatmap image")
                heatmap_error = (
                    f"Heatmap image: {type(exc).__name__}: {exc}"
                    if settings.DEBUG else "The attention image could not be prepared."
                )

        response_data["heatmap_data_url"] = heatmap_url
        response_data["heatmap_url"] = heatmap_url
        response_data["heatmap_overlay_url"] = overlay_url
        response_data["heatmap_method"] = "Grad-CAM" if heatmap_url else None
        response_data["heatmap_error"] = heatmap_error
        # Explain the stage-one decision using the actual calculated distance,
        # not an invented probability. Exposed only for the current report.
        response_data["gate_nearest_class"] = result.get("gate_nearest_class")
        response_data["gate_distance"] = result.get("gate_distance")
        response_data["gate_threshold"] = result.get("gate_threshold")
        response_data["classifier_key"] = result.get("classifier_key")
        return Response(response_data)