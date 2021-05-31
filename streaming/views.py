from account.models import CustomUser
from django.shortcuts import redirect, render,get_object_or_404
from django.contrib.auth import authenticate, get_user_model,logout,login as auth_login
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
import time
import cv2
from django.views.decorators import gzip
import threading
from django.http import HttpResponse,StreamingHttpResponse
from django.utils import timezone
from django.contrib.auth.decorators import login_required
# Create your views here.
id=0
stream_list=[]
limit=3
mapping={}
def streaming(request,username):
    
    user=get_user_model().objects.get(pk=username)
    if request.user.username==username :
        user.video.create=timezone.now()
        user.video.streaming=True
        user.video.save()
    else:
        user.video.views+=1
        user.video.save()
    return render(request,'stream.html',{'streamer':user,'watcher':request.user})


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.thread=threading.Thread(target=self.update, args=())
        self.flag=False
        self.fps=0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        image=cv2.flip(image,1)
        cv2.putText(image,'{}'.format(self.fps),(20,23),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,255,0))
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        prev=0
        while True:
            (self.grabbed, self.frame) = self.video.read()
            if self.grabbed==False:
                break
            cur=time.time()
            sec=cur-prev
            prev=cur
            if sec<=0.0:
                self.fps='very fast'
            else:
                self.fps='{:.2f}fps'.format(1/(sec))

    def gen(self):
        while True:
            if self.flag:
                break
            frame = self.get_frame()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')            
            
            

        
@gzip.gzip_page
def video_feed(request,username):
    global id,stream_list,limit,mapping
    
    if request.user.username==username :
        if id+1>=limit:
            return render(request,'error.html')
        try:
            print('bring cameara number : ',id)
            cam = VideoCamera()
            print('append camera object . . . .')
            stream_list.append(cam)
            mapping[username]=id
            stream_list[id].thread.start()
            id+=1
            
            
        except:  # This is bad! replace it with proper handling
            print('bad network')
    elif not get_user_model().objects.get(pk=username).video.streaming:
        print('방송중이 아닙니다.')
        return redirect('home')

    return StreamingHttpResponse(stream_list[mapping[username]].gen(), content_type="multipart/x-mixed-replace;boundary=frame")


def end_broadcast(request,username):
    global stream_list,id,mapping
    stream_list[mapping[username]].flag=False
    time.sleep(1.2)
    stream_list[mapping[username]].video.release()
    del stream_list[mapping[username]]

    user=get_user_model().objects.get(pk=username)
    #update related tables 
    user.total.total_views+=user.video.views
    user.video.views=0
    user.total.total_like+=user.video.like
    user.video.like=0
    user.total.total_bad+=user.video.bad
    user.video.bad=0
    user.video.streaming=False
    user.video.save()
    user.total.save()
    print('exit success')
    id-=1
    print('id after give it back : ',id)
    return redirect('home')
            

