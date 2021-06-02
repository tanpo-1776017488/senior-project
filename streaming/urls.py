from django.contrib import admin
from django.contrib.auth import views
from django.urls import path,include
from .views import *
from . import views
app_name='streaming'
urlpatterns = [
    path('<str:username>/',streaming,name='streaming'),
    path('<str:username>/video/',video_feed,name='video'),
    path('<str:username>/end/',end_broadcast,name='end'),
    path('like/', views.video_like, name='video_like'), #url생성
]
