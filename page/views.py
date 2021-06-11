from django.shortcuts import render
from account.models import video
def home(request):
    stream_info=video.objects.filter(streaming=True)   #false는 테스트 용도로 설정해 둠. return object가 없어도 정상적으로 작동함.
    return render(request,'home.html',{'video_info':stream_info})



# Create your views here.


