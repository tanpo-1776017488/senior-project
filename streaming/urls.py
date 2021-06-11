from django.contrib import admin
from django.urls import path,include
from .views import *
app_name='streaming'
urlpatterns = [
    path('<str:username>/',streaming,name='streaming'),
    path('<str:username>/video/',video_feed,name='video'),
    path('<str:username>/end/',end_broadcast,name='end'),
    path('<str:username>/upload/',upload_image,name='upload_image'),
    path('<str:username>/like/',video_like,name='video_like'),
    path('<str:username>/facebank/',upload_facebank_img,name='face'),
    path('<str:username>/bad/',video_bad,name='video_bad'),
]
