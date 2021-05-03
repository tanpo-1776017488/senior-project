import cv2
import numpy as np
import path
import os
from imutils import paths
# def load_data(inden,prefix_path):
#     faces_name=[]
#     label_name=[]
#     if not os.path.isfile(inden):
#         exit(-1)
#     with open(inden,'r') as f:
#         for line in f:
#             name,labeled=line.split()
#             faces_name.append(name)
#             label_name.append(int(labeled))
#             #print(name," ",labeled)
#     faces=[]
    
#     for i in range(len(faces_name)):
#         img_path=os.path.join(prefix_path,faces_name[i])
#         img=cv2.imread(img_path)
#         if img is None:
#             print('fail to open ',faces_name[i])
#             continue
#         faces.append(img)
#     faces=np.array(faces)
#     label_name=np.array(label_name)
#     return (faces,label_name)
    

    


def load_data(inputPath,minSamples=5):
	imagePaths = list(paths.list_images(inputPath))
	names = [p.split(os.path.sep)[-2] for p in imagePaths]
	(names, counts) = np.unique(names, return_counts=True)
	names = names.tolist()
	# labels
	faces = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
		name = imagePath.split(os.path.sep)[-2]
        
		
        
		#안쓸예정
		if counts[names.index(name)] < minSamples:
			continue

		# perform face detection
		
		faces.append(image)
		labels.append(int(name))

	# convert our faces and labels lists to numpy arrays
	faces = np.array(faces)
	labels = np.array(labels)

	# return a 2-tuple of the faces and labels
	return (faces, labels)

if __name__=="__main__":
	load_data('C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/dst',minSamples=5)