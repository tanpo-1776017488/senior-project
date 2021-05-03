import cv2
import numpy as np

def clear_label(ori_label_path,label_path,wr_path):
    ff=open(label_path,'r')
    rm_list=[]
    for name in ff.readlines():
        rm_list.append(name[:-1])
    ff.close()
    ori_img=[]
    ori_label=[]
    dd=open(ori_label_path,'r')
    for line in dd.readlines():
        name,label=line.split()
        ori_img.append(name)
        ori_label.append(label)

    for name in rm_list:
        if name in ori_img:
            idx=ori_img.index(name)
            ori_img.pop(idx)
            ori_label.pop(idx)
    dd.close()

    with open(wr_path,'w') as fo:
        for i in range(len(ori_img)):
            fo.write(str(ori_img[i])+' '+str(ori_label[i])+'\n')
    
        
    
if __name__=='__main__':
    src_path='remove.txt'
    ori_path='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/Anno/identity_CelebA.txt'
    wr_path='label.txt'
    clear_label(ori_path,src_path,wr_path)

