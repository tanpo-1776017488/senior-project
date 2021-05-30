from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import CustomUser

#UserCreationForm
# 장고에서 기본적으로 제공하는 회원가입 폼
# attr : username, password1,password2(confirm password)

class UserForm(UserCreationForm):
    email=forms.EmailField(label="email")
    name=forms.CharField(label='name')
    img=forms.ImageField(required=False,label='img')
    class Meta:
        model=CustomUser
        fields=("username","email","name","img")
        
        