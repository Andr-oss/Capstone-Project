from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('new-videos/', views.new_videos_view, name='new_videos'),
    path('process-video/', views.process_video, name='process_video'),
    path('stop-video/', views.stop_video, name='stop_video'),
    path('get-progress/', views.get_progress, name='get_progress'),  # Add this line
    path('livestream/', views.livestream_view, name='livestream'),
    path('download-final-csv/', views.download_final_csv, name='download_final_csv'),
    path('download/<str:filename>', views.download_file, name='download_file'),

]