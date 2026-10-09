# backend/detector/views.py
# Source: https://www.django-rest-framework.org/api-guide/views/

import base64
import io
from PIL import Image
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer
from django.views.generic import TemplateView


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
        result = predict_image(image_file)   # inference reads in memory, nothing persisted

        prediction = Prediction(
            label=result["label"],
            confidence=result["confidence"],
            is_known_disease=result["is_known_disease"],
            is_banana_leaf=result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
        )
        prediction.save()

        # Attach the heatmap as a base64 data URL — nothing written to disk
        prediction.heatmap_data_url = ""
        if result["heatmap_array"] is not None:
            heatmap_img = Image.fromarray((result["heatmap_array"] * 255).astype("uint8"))
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            prediction.heatmap_data_url = (
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )

        response_data = PredictionSerializer(prediction).data
        response_data["heatmap_data_url"] = prediction.heatmap_data_url
        return Response(response_data)