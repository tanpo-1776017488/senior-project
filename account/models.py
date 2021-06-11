from django.db import models
from django.contrib.auth.models import AbstractUser
import datetime
import PIL
from django.db.models.fields import BLANK_CHOICE_DASH, related
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

# tables
def post_image_path(instance,filename):
    
    return f'account/{instance.username}/profile.jpg'

def facebank_image(instance,filename):
    filename=filename[:-4]
    number=str(instance.global_count)
    return f'account/{instance.username}/facebank/{number}/{filename}.jpg'

class CustomUser(AbstractUser):
    name=models.CharField(max_length=10,default='admin')
    username=models.CharField(max_length=15,primary_key=True)
    #img=models.ImageField(blank=True,null=True,default='default.png')
    img=ProcessedImageField(
        default='default.png',
        upload_to=post_image_path,
        processors=[ResizeToFill(300,300)],
        format='JPEG',
        options={'quality':100}
    )
    face_bank=ProcessedImageField(
        blank=True,
        null=True,
        upload_to=facebank_image,
        processors=[ResizeToFill(300,300)],
        format='JPEG',
        options={'quality':100}
    )
    global_count=models.IntegerField(default=0,null=True,blank=True)
    pub_date=models.DateTimeField(default=timezone.now,editable=False,blank=True,null=True)
    nickname=models.CharField(max_length=20,blank=True,null=True,unique=True)


class profile(models.Model):
    user=models.OneToOneField(CustomUser,on_delete=models.CASCADE)
    sub=models.IntegerField(blank=True,default=0)
    #img=models.ImageField(blank=True,null=True,default='default.png',upload_to='account/{}'.format(user.username))

class video(models.Model):
    user=models.OneToOneField(CustomUser,on_delete=models.CASCADE)
    like=models.IntegerField(blank=True,default=0)
    bad=models.IntegerField(blank=True,default=0)
    streaming=models.BooleanField(blank=True,null=True,default=False)
    title=models.CharField(max_length=20,blank=True,null=True)
    create=models.DateTimeField(editable=True,blank=True,null=True)
    views=models.IntegerField(default=0,blank=True)

    def count_likes(self):
        return self.like
    
    def __str__(self):
        return self.title
    
    def count_bad(self):
        return self.bad

    
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
