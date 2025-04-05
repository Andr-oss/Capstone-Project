from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('process/', views.process_view, name='process'),
    path('process-video/', views.process_video, name='process_video'),
    path('stop-video/', views.stop_video, name='stop_video'),
    path('download-csv/', views.download_csv, name='download_csv'),
    path('livestream/', views.livestream_view, name='livestream'),
    path('visualize/', views.visualize_view, name='visualize'),
    path('run-visualize/', views.run_visualize, name='run_visualize'),
]

