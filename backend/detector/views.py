# backend/detector/views.py
# Source: https://www.django-rest-framework.org/api-guide/views/

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer
from django.views.generic import TemplateView
from django.conf import settings
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)


def _image_preview_data_url(image):
    """A compact preview of the image used for the Grad-CAM pass."""
    preview = image.convert("RGB").copy()
    preview.thumbnail((900, 900), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class UploadPageView(TemplateView):
    template_name = "upload.html"


class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if image_file is None:
            return Response({"error": "Select an image first."}, status=400)
        # Load heavy AI dependencies only when a scan is requested.
        from .inference import predict_image
        # Always predict from the original photo. Foreground isolation is
        # applied automatically inside predict_image only for Grad-CAM.
        result = predict_image(image_file)

        prediction = Prediction(
            label=result["label"],
            confidence=result["confidence"],
            is_known_disease=result["is_known_disease"],
            is_banana_leaf=result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
        )
        prediction.save()

        # Keep grayscale heatmap for the existing interactive viewer, while
        # also exposing the blended overlay from the earlier Django project.
        heatmap_url = None
        overlay_url = None
        heatmap_error = result.get("heatmap_error")
        if result["heatmap_array"] is not None:
            try:
                from .heatmap import make_heatmap_urls
                source = result.get("heatmap_source_image")
                if source is None:
                    raise ValueError("No image available for heatmap generation")
                heatmap_url, overlay_url = make_heatmap_urls(
                    source,
                    result["heatmap_array"],
                    foreground_mask=result.get("heatmap_foreground_mask"),
                )
            except Exception as exc:
                logger.exception("Failed to encode a Grad-CAM heatmap image")
                heatmap_error = (
                    f"Heatmap image: {type(exc).__name__}: {exc}"
                    if settings.DEBUG else "The attention image could not be prepared."
                )

        removed_preview = None
        if result.get("background_removed_applied"):
            try:
                removed_preview = _image_preview_data_url(result["heatmap_source_image"])
            except Exception:
                logger.exception("Could not prepare background-removed image preview")

        response_data = PredictionSerializer(prediction).data
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
        response_data["background_removed_applied"] = result.get("background_removed_applied", False)
        response_data["background_removal_method"] = result.get("background_removal_method")
        response_data["background_removal_reason"] = result.get("background_removal_reason")
        response_data["background_removed_preview_url"] = removed_preview
        return Response(response_data)