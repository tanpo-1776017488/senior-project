from django.db import models
from django.contrib.auth.models import AbstractUser
import datetime
import PIL
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


# tables
class CustomUser(AbstractUser):
    name=models.CharField(max_length=10,default='admin')
    username=models.CharField(max_length=15,primary_key=True)
    img=models.ImageField(blank=True,null=True,upload_to='account/',default='default.png')
    pub_date=models.DateTimeField(default=timezone.now,editable=False,blank=True,null=True)
    nickname=models.CharField(max_length=20,blank=True,null=True,unique=True)


class profile(models.Model):
    user=models.OneToOneField(CustomUser,on_delete=models.CASCADE)
    sub=models.IntegerField(blank=True,default=0)

class video(models.Model):
    user=models.OneToOneField(CustomUser,on_delete=models.CASCADE)
    like=models.IntegerField(blank=True,default=0)
    bad=models.IntegerField(blank=True,default=0)
    streaming=models.BooleanField(blank=True,null=True,default=False)
    title=models.CharField(max_length=20,blank=True,null=True)
    create=models.DateTimeField(editable=True,blank=True,null=True)
    views=models.IntegerField(default=0,blank=True)
    
class total(models.Model):
    user=models.OneToOneField(CustomUser,on_delete=models.CASCADE)
    total_views=models.IntegerField(blank=True,default=0)
    total_like=models.IntegerField(blank=True,default=0)
    total_bad=models.IntegerField(blank=True,default=0)


#if CustomUser created, automatically create other tables
@receiver(post_save,sender=CustomUser)
def create_user_profile(sender,instance,created,**kwargs):
    if created:
        profile.objects.create(user=instance)

@receiver(post_save,sender=CustomUser)
def save_user_profile(sender,instance,**kwargs):
    instance.profile.save()

@receiver(post_save,sender=CustomUser)
def create_user_total(sender,instance,created,**kwargs):
    if created:
        total.objects.create(user=instance)

@receiver(post_save,sender=CustomUser)
def save_user_total(sender,instance,**kwargs):
    instance.total.save()


@receiver(post_save,sender=CustomUser)
def create_user_video(sender,instance,created,**kwargs):
    if created:
        video.objects.create(user=instance)

@receiver(post_save,sender=CustomUser)
def save_user_video(sender,instance,**kwargs):
    instance.video.save()
