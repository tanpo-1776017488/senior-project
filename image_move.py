import os
import shutil

pre_path='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/img_align_celeba'
make_dir='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/color_dst'

with open('label.txt','r') as ff:
    for line in ff:
        name,identity=line.split()
        img_path=os.path.join(pre_path,name)
        dst_path=os.path.join(make_dir,identity)
        dst_path=os.path.join(dst_path,name)
        shutil.move(img_path,dst_path)


