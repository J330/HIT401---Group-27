from django.urls import path

from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('result/', views.result, name='result'),
    path('api/predict/', views.predict_api, name='predict_api'),
]
