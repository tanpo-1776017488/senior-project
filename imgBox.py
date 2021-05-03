from imutils import paths
import numpy as np
import cv2
import os
import face_recognition

prototxtPath = 'C:/Users/sorjt/Desktop/mylab/experiment/face_detector/deploy.prototxt'
weightsPath = 'C:/Users/sorjt/Desktop/mylab/experiment/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
des_path='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/only_face'


remove_file='remove.txt'
with open('C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/Anno/identity_CelebA.txt','r') as rFile:
    with open(remove_file,'w') as rm:
        pre_addr='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/img_align_celeba'
        for line in rFile:
            name,label=line.split()
            img_path=os.path.join(pre_addr,name)
            img=cv2.imread(img_path)
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb,model='hog')
            if len(boxes)==0 or img is None:
                print('remove ',name)
                rm.write(str(name)+'\n')
                continue
            try:
                for (top,right,bottom,left) in boxes:
                    faceROI=img[top:bottom,left:right]
                    faceROI=cv2.resize(faceROI,(47,62))
                    faceROI=cv2.cvtColor(faceROI,cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(des_path,name),faceROI)
            except:
                print('remove ',name)
                rm.write(str(name)+'\n')
