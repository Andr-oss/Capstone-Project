# rodent_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('tracking.urls')),  # Direct root URL to the tracking app's urls.py
]
