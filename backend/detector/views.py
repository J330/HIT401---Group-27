# backend/detector/views.py
# Source: https://www.django-rest-framework.org/api-guide/views/

import io
from PIL import Image
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.conf import settings
from django.shortcuts import redirect
from django.middleware.csrf import get_token
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.authtoken.models import Token
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from .models import Prediction
from .serializers import PredictionSerializer
from .inference import predict_image


def csrf_token(request):
    return JsonResponse({"csrfToken": get_token(request)})


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        default_username = settings.DEFAULT_ACCOUNT_USERNAME
        default_password = settings.DEFAULT_ACCOUNT_PASSWORD
        User.objects.get_or_create(username=default_username, defaults={"is_active": True})
        account = User.objects.get(username=default_username)
        account.set_password(default_password)
        account.save(update_fields=["password"])

        user = authenticate(
            username=request.data.get("username", ""),
            password=request.data.get("password", ""),
        )
        if user is None:
            return Response({"detail": "Invalid username or password."}, status=400)

        login(request, user)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "username": user.username})



class PredictView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        image_file = request.FILES["image"]
        result = predict_image(image_file)

        prediction = Prediction(
            user=request.user if request.user.is_authenticated else None,
            image=image_file,
            label=result["label"],
            confidence=result["confidence"],
            is_known_disease=result["is_known_disease"],
            is_banana_leaf=result["is_banana_leaf"] if result["is_banana_leaf"] is not None else True,
        )

        # Only authenticated scans are persisted for dashboard history.
        if request.user.is_authenticated:
            if result["heatmap_array"] is not None:
                heatmap_img = Image.fromarray((result["heatmap_array"] * 255).astype("uint8"))
                buffer = io.BytesIO()
                heatmap_img.save(buffer, format="PNG")
                prediction.heatmap.save("heatmap.png", ContentFile(buffer.getvalue()), save=False)

            prediction.save()
            return Response(PredictionSerializer(prediction).data)

        return Response({
            "id": None,
            "image": None,
            "label": prediction.label,
            "confidence": prediction.confidence,
            "is_known_disease": prediction.is_known_disease,
            "is_banana_leaf": prediction.is_banana_leaf,
            "heatmap": None,
            "created_at": None,
        })


# Returns all saved prediction results, newest first.
class PredictionListView(ListAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user).order_by("-created_at")


class PredictionDetailView(RetrieveUpdateDestroyAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("http://127.0.0.1:8080/account.html")

    predictions = list(Prediction.objects.filter(user=request.user).order_by("-created_at"))
    context = {
        "predictions": predictions,
        "total_scans": len(predictions),
        "healthy_scans": sum(prediction.label == "healthy" for prediction in predictions),
        "diseased_scans": sum(prediction.label == "black_sigatoka" for prediction in predictions),
    }
    return render(request, "dashboard.html", context)
