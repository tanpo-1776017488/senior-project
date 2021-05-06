import cv2
import os
cur_path=os.path.join(os.getcwd(),'video')

folder=os.listdir(cur_path)
for name in folder:
    path=os.path.join(cur_path,name)
    print(path)
    cnt=0
    if os.path.isdir(path):
        video=os.listdir(path)[0]
        print(video)
        cap=cv2.VideoCapture(os.path.join(path,video))
        while True:
            ret,frame=cap.read()
            if ret is False or frame is None:
                break
            cv2.imwrite('{}'.format(os.path.join(path,str(cnt)+'.jpg')), frame)
            cnt+=1


