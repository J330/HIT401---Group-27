from django.urls import path

from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('result/', views.result, name='result'),
    path('about/', views.about, name='about'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/delete/<int:scan_id>/', views.delete_scan, name='delete_scan'),
    path('api/predict/', views.predict_api, name='predict_api'),
]
