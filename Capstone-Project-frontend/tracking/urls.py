from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('process/', views.process_view, name='process'),
    path('livestream/', views.livestream_view, name='livestream'),
    path('visualize/', views.visualize_view, name='visualize'),
    path('run-visualize/', views.run_visualize, name='run_visualize'),
    path('process-video/', views.process_video, name='process_video'),
    path('check-task-status/', views.check_task_status, name='check_task_status'),
    path('download-result/', views.download_result, name='download_result'),
]

