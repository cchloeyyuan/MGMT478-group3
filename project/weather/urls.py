"""
URL configuration for weather project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from worldmap.views import map_view
from worldmap.views import map_request
from worldmap.views import time_period_request
from django.urls import path
from worldmap import views

urlpatterns = [
    path('map/', map_view, name='map_view'),
    path('admin/', admin.site.urls),
    path('map_request/', map_request, name='map_request'),
    path('contact/', views.contact, name='contact'),
    path('time-period_request/', views.time_period_request, name='time_period_request'),
    path('', views.update_predictions_request, name='update_predictions_request')
]

