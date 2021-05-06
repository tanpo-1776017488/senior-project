import cv2
import os
import shutil
path='./dataset/unknown'
for name in os.listdir(path):
    img_path=os.path.join(path,name)
    img=cv2.imread(img_path)
    if img.shape[0] <100 and img.shape[1] <100:
        print('rm : {} width : {}, height : {}'.format(os.path.basename(img_path),img.shape[1],img.shape[0]))
        os.remove(img_path)
