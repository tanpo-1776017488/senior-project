from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
import load_celeba_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imutils import build_montages

import numpy as np
import argparse
import imutils
import time
import cv2
import os

# input argument
args=dict()
args["input"]='C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/dst_1'
args["classifier"]='svm'
args["num_components"]=150

# load the image dataset
print("[INFO] loading dataset...")
(faces, labels) =load_celeba_data.load_data(args["input"])
print("[INFO] {} images in dataset".format(len(faces)))

# flatten 2d data into 1D data
pcaFaces = np.array([f.flatten() for f in faces])

# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.25,stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split

# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(svd_solver="randomized",n_components=args["num_components"],whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(end - start))


#model selection
if args["classifier"]=='svm':
    model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
elif args["classifier"]=='knn':
    model=KNeighborsClassifier(n_neighbors=3,weights="distance")
elif args["classifier"]=='lda':
    model=LinearDiscriminantAnalysis()
else:
    print('input correct classifier!')
    exit(-1)
# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print('{} prediction :'.format(args["classifier"]))
print(classification_report(testY, predictions))