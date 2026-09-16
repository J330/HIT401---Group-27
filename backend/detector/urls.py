# backend/detector/urls.py

from django.urls import path
from .views import csrf_token, LoginView, PredictView, PredictionListView, PredictionDetailView
from .api_docs import ApiDocsView

urlpatterns = [
    path("csrf/", csrf_token),
    path("login/", LoginView.as_view()),
    path("predict/", PredictView.as_view()),
    path("predictions/", PredictionListView.as_view()),
    path("predictions/<int:pk>/", PredictionDetailView.as_view()),
    path("docs/", ApiDocsView.as_view()),
]
