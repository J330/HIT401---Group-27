# backend/detector/api_docs.py
# A tiny, dependency-free API description endpoint at GET /api/docs/ -- not
# a full OpenAPI spec, just enough for anyone hitting the URL to see what's
# available without reading the code.
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings


class ApiDocsView(APIView):
    def get(self, request):
        return Response(
            {
                "active_model": settings.ACTIVE_MODEL,
                "endpoints": {
                    "POST /api/predict/": {
                        "description": "Upload a banana leaf photo and get a prediction.",
                        "body": "multipart/form-data with an 'image' file field",
                        "returns": {
                            "label": "'healthy' | 'black_sigatoka' | 'not_black_sigatoka'",
                            "confidence": "float 0-1",
                            "is_known_disease": "bool",
                            "is_banana_leaf": "bool or null",
                            "heatmap": "URL to the explainability image, or null",
                        },
                    }
                },
            }
        )
