
from django.contrib import admin
from django.urls import path,include
from django.contrib.auth import views as auth_views
from .views import *

app_name='account'
urlpatterns = [
    path('login/',auth_views.LoginView.as_view(template_name='login.html'),name='login'),
    #path('login/',login,name='login'),
    path('sign_up/',sign_up,name='sign_up'),
    path('logout/',auth_views.LogoutView.as_view(),name='logout'),
    path('profile/<str:username>',profile,name='profile'),
   
]