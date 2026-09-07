# backend/detector/views.py
# Source: https://www.django-rest-framework.org/api-guide/views/

import io
from PIL import Image
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer
from .inference import predict_image


class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES["image"]
        result = predict_image(image_file)

        prediction = Prediction(
            image=image_file,
            label=result["label"],
            confidence=result["confidence"],
            is_known_disease=result["is_known_disease"],
            is_banana_leaf=result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
        )

        # Turn the Grad-CAM array into a saved PNG, if there is one
        if result["heatmap_array"] is not None:
            heatmap_img = Image.fromarray((result["heatmap_array"] * 255).astype("uint8"))
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            prediction.heatmap.save("heatmap.png", ContentFile(buffer.getvalue()), save=False)

        prediction.save()
        return Response(PredictionSerializer(prediction).data)
