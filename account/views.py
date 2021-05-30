from account.models import *
from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth import authenticate, get_user_model,logout,login as auth_login
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from .forms import UserForm



def sign_up(request):
    # POST인 경우 새로운 사용자 생성
    # GET인 경우 return sin-up.html
    if request.method=="POST":
        form=UserForm(request.POST,request.FILES)
        #print(form)
        if form.is_valid():
            print('not entered')
            #tmp_user=form.save(commit=False)
            form.save()
            #print(request.FILES['img'])
            #tmp_user.img=request.FILES['img']
            #tmp_user.save()
            #tmp_user.img=
            username=form.cleaned_data.get('username')
            password=form.cleaned_data.get('password1')
            user=authenticate(username=username,password=password)
            auth_login(request,user)
        return redirect('home')
    
    else :
        form=UserForm()
    return render(request,'sin-up.html',{'form':form})

def profile(request,username):
    user=get_object_or_404(get_user_model(),pk=username)
    return render(request,'profile.html',{'Cuser':user})

    