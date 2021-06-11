from types import prepare_class
from MTCNN.MTCNN import mtcnn_custom
from torch._C import device
from account.models import CustomUser, facebank_image
from django.shortcuts import redirect, render,get_object_or_404
from django.contrib.auth import authenticate, get_user, get_user_model,logout,login as auth_login
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from django.views.decorators import gzip
from django.http import HttpResponse,StreamingHttpResponse
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import threading
import time
import cv2
import json
from django.views.decorators.http import require_POST
# mobilefacenet package below
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
print(sys.path[0])
import argparse
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
import numpy as np
#from utils.util import *
from utils.align_trans import *
from MTCNN.MTCNN import mtcnn_custom
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import stream.settings
import os
# Create your views here.
id=0
stream_list=[]
limit=3
mapping={}

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized
def upload_image(request,username):
    file=request.FILES['file1']
    filename=file._name
    dir_name='account/{}/facebank'.format(username)
    path=os.path.join(stream.settings.MEDIA_ROOT,dir_name)
    
    if not os.path.exists(path):
        os.mkdir(path)

    path=os.path.join(path,filename[:-3])
    if not os.path.exists(path):
        os.mkdir(path)
    #print('file name:',filename)
    path=os.path.join(path,filename)
    fp=open(path,'wb')
    for chunk in file.chunks():
        fp.write(chunk)
    fp.close()
    return redirect('streaming:streaming',username)


def streaming(request,username):
    
    user=get_user_model().objects.get(pk=username)
    if request.user.username==username :
        user.video.create=timezone.now()
        #user.video.streaming=True
        #user.video.save()
    else:
        user.video.views+=1
        user.video.save()
    return render(request,'stream.html',{'streamer':user,'watcher':request.user})


class VideoCamera(object):
    def __init__(self,username='admin',threshold=80,update_val=True,tta=False,c=True,scale=0.3,min_face=20,embedding=512,bank_path='media/account/facebank',model_path='Weights/model_final.pth'):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.thread=threading.Thread(target=self.update, args=())
        self.flag=True
        # recognition parameter
        self.threshold=threshold
        self.tta=tta
        self.score=c
        self.scale=scale
        self.min_face=min_face
        self.embedding=embedding
        self.facebank='media/account/{}/facebank'.format(username)
        self.model_path=model_path
        self.up=update_val
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('using ',self.device)
        self.face_detector=mtcnn_custom(device=self.device,p_model_path='MTCNN/weights/pnet_Weights',r_model_path='MTCNN/weights/rnet_Weights',o_model_path='MTCNN/weights/onet_Weights')
        print('face detector created...')

        #prepare pretrained model
        self.detect_model = MobileFaceNet(self.embedding).to(self.device)  # embeding size is 512 (feature vector)
        # self.check_point=torch.load('Weights/model_final_t.pth',map_location=lambda storage, loc: storage)
        # self.detect_model.load_state_dict(self.check_point['model_state_dict'])
        self.detect_model.load_state_dict(torch.load('Weights/model_final.pth',map_location=lambda storage, loc: storage))
        print('MobileFaceNet face detection model generated')
        self.detect_model.eval()

        #face bank update
        if self.up:
            self.targets, self.names = prepare_facebank(self.detect_model, path=self.facebank, tta=self.tta)
            print('facebank updated')
        else:
            self.targets, self.names = load_facebank(path=self.facebank)
            print('facebank loaded')
            # targets: number of candidate x 512
        

    def __del__(self):
        self.video.release()
    
    def update_facebank(self,img_list):
        for img_path in img_list:
            
            img=cv2.imread(img_path)
            bboxes,landmarks=self.face_detector.detect_all_net(image=img,mini_face=self.min_face)
            faces= Face_alignment(img,default_square=True,landmarks=landmarks)
            try:
                os.remove(img_path)
            except:
                print('fail to remove')
            cv2.imwrite(img_path,faces[0])
        self.targets,self.names=prepare_facebank(self.detect_model, path=self.facebank, tta=self.tta)
        print('new facebank uploaded !!...')

    def get_frame(self): #여기다가 face detection, recognition기능을 넣으면 문제없음.
        
        frame = self.frame #thread가 update하는 이미지를 가져 옴.
        
        if frame is not None and self.flag is True:
            try:
                start_time=time.time()
                input=resize_image(frame,self.scale)# input size를 줄여줌으로 speed up 가능
                #print('get bboxes')
                # bboxes, landmarks = create_mtcnn_net(input, self.min_face, self.device, p_model_path='MTCNN/weights/pnet_Weights',
                #                                             r_model_path='MTCNN/weights/rnet_Weights',
                #                                             o_model_path='MTCNN/weights/onet_Weights')
                #print('sucess bbox')
                
                bboxes,landmarks=self.face_detector.detect_all_net(image=input,mini_face=self.min_face)
                
                if bboxes != []:
                    bboxes=bboxes/self.scale
                    landmarks=landmarks/self.scale

                faces= Face_alignment(frame,default_square=True,landmarks=landmarks)

                

                embs=[]
                test_transform = trans.Compose([
                                            trans.ToTensor(),
                                            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

                for img in faces:
                    if self.tta:
                        mirror = cv2.flip(img,1)
                        emb = self.detect_model(test_transform(img).to(self.device).unsqueeze(0))
                        emb_mirror = self.detect_model(test_transform(mirror).to(self.device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(self.detect_model(test_transform(img).to(self.device).unsqueeze(0)))
                    

                source_embs=torch.cat(embs)
                diff=source_embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0) # i.e
                dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
                minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
                min_idx[minimum > ((self.threshold-156)/(-80))] = -1  # if no match, set idx to -1
                score = minimum
                results = min_idx
                score_100 = torch.clamp(score*-80+156,0,100)
                FPS=1.0/(time.time()-start_time)
                
                cv2.putText(frame,'FPS : {:.1f}'.format(FPS),(10,15),cv2.FONT_HERSHEY_DUPLEX,0.75,(255,0,255))
                for i,b in enumerate(bboxes):
                    b=b.astype('uint32')
                    cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),1)
                    try:
                        if self.names[results[i]+1]=='Unknown': #mosic func
                            #print('detect unknwon')
                            face_region=frame[b[1]:b[3],b[0]:b[2]]
                            face_region=cv2.blur(face_region,(30,30))
                            frame[b[1]:b[3],b[0]:b[2]]=face_region

                    except:
                        pass
                    # development version
                    # if self.score:
                    #     cv2.putText(frame,self.names[results[i]+1]+' score:{:.0f}'.format(score_100[i]),(int(b[0]),int(b[1]-25)),cv2.FONT_ITALIC,1,(255,255,0))
                    # else:
                    #     cv2.putText(frame,self.names[results[i]+1],(int(b[0]),int(b[1]-25)),cv2.FONT_ITALIC,1,(255,255,0))
                    
            except:
                pass

            _, jpeg = cv2.imencode('.jpg',frame)
            return jpeg.tobytes()
            
            

    def update(self): # thread를 사용하여 지속적으로 cap.read반복
        
        while True:
            (self.grabbed, self.frame) = self.video.read()
            if self.grabbed==False or self.flag==False:
                break

    def gen(self):  
        while True:
            if self.flag==False:
                break

            frame = self.get_frame()

            if frame is None:
                continue
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')            
                 
@gzip.gzip_page
def video_feed(request,username):
    global id,stream_list,limit,mapping
    user=get_user_model().objects.get(pk=username)
    print('is streaming ? ',user.video.streaming)
    if request.user.username==username :
        
        if not user.video.streaming:
            if id+1>=limit:
                return render(request,'error.html')
            try:
                print('bring cameara number : ',id)
                cam = VideoCamera(username=username)
                print('append camera object . . . .')
                stream_list.append(cam)
                mapping[username]=id
                stream_list[id].thread.start()
                user.video.streaming=True
                user.video.save()                
                id+=1              
            except:  # This is bad! replace it with proper handling
                print('bad network')

    elif not get_user_model().objects.get(pk=username).video.streaming:
        print('방송중이 아닙니다.')
        return redirect('home')
    user.video.views+=1
    user.video.save()
    return StreamingHttpResponse(stream_list[mapping[username]].gen(), content_type="multipart/x-mixed-replace;boundary=frame")
    


def end_broadcast(request,username):
    global stream_list,id,mapping
    
    try:
        stream_list[mapping[username]].flag=False
        time.sleep(1.2)
        stream_list[mapping[username]].video.release()
        del stream_list[mapping[username]]
        print('return wep cam')
    except:
        pass
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

@require_POST 
def video_like(request,username):
    pk=request.POST.get('pk',None)
    print('pk : ',pk)
    user=get_user_model().objects.get(pk=pk)
    like=0
    message='bad'
    if user is not None:
        message='good'
        print('sucess getting user !')
        user.video.like+=1
        user.video.save()
        like=user.video.like

    context={'likes_count':user.video.like,'messae':message}
    return HttpResponse(json.dumps(context),content_type='application/json')


@require_POST 
def video_bad(request,username):
    pk=request.POST.get('pk',None)
    print('pk : ',pk)
    user=get_user_model().objects.get(pk=pk)
    bad=0
    message='bad'
    if user is not None:
        message='good'
        print('sucess getting user !')
        user.video.bad+=1
        user.video.save()
        bad=user.video.bad

    context={'bad_count':user.video.bad,'messae':message}
    return HttpResponse(json.dumps(context),content_type='application/json')

def upload_facebank_img(request,username):
    print('method',request.method)
    img_list=[]
    if request.method=='POST':
        user=get_user_model().objects.get(pk=username)
        for img in request.FILES.getlist('imgs'):
            user.face_bank=img
            user.save()
            img_list.append(user.face_bank.path)

        stream_list[mapping[username]].update_facebank(img_list)
        user.global_count+=1
        user.save()
    return redirect('streaming:streaming',username=username)
