# backend/detector/urls.py

from django.urls import path
from .views import PredictView
from .api_docs import ApiDocsView

urlpatterns = [
    path("predict/", PredictView.as_view()),
    path("docs/", ApiDocsView.as_view()),
]
