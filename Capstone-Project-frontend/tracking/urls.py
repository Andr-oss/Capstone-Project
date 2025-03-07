# tracking/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('new-videos/', views.new_videos_view, name='new_videos'),
    path('register/', views.register_view, name='register'),
    path('download-csv/', views.download_csv, name='download_csv'),
    path('livestream/', views.livestream_view, name='livestream'),
]
