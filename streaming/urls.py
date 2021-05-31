from django.contrib import admin
from django.urls import path,include
from .views import *
app_name='streaming'
urlpatterns = [
    path('<str:username>/',streaming,name='streaming'),
    path('<str:username>/video/',video_feed,name='video'),
    path('<str:username>/end/',end_broadcast,name='end'),
]
