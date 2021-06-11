from django import forms
from django.contrib.auth.forms import UserCreationForm, UsernameField
from django.contrib.auth.models import User
from .models import CustomUser

#UserCreationForm
# 장고에서 기본적으로 제공하는 회원가입 폼
# attr : username, password1,password2(confirm password)

class UserForm(UserCreationForm):
    email=forms.EmailField(label="email",required=True)
    name=forms.CharField(label='Name',required=True)
    img=forms.ImageField(required=False,label='image')
    username=forms.CharField(required=True,label='ID',max_length=15)
    nickname=forms.CharField(required=True,label='nickname',max_length=20)
    class Meta:
        model=CustomUser
        fields=("name","username","password1","password2","nickname","email","img")
        
        