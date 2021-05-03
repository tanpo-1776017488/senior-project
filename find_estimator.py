# USAGE
# python find_estimator.py --input caltech_faces -m svm

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from face import load_face_dataset
import load_celeba_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from imutils import build_montages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# input argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,help="path to input directory of images")
ap.add_argument("-f", "--face", type=str,default="face_detector",help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=150,help="# of principal components")
ap.add_argument("-m","--classifier",type=str,default='svm')
args = vars(ap.parse_args())

# load face detector
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the image dataset
print("[INFO] loading dataset...")
(faces, labels) =  load_face_dataset(args["input"], net,minConfidence=0.8, minSamples=5)
print("[INFO] {} images in dataset".format(len(faces)))

# flatten 2d data into 1D data
pcaFaces = np.array([f.flatten() for f in faces])

# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

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
    param_grid = {'C': [1,10,100,1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
              'kernel':['linear','poly','rbf','sigmoid'],
              'degree':[1,2,3,4,5],
              'gamma':['scale','auto',0.001,0.01,0.1],
              'class_weight':[None,'balanced']}
    model = GridSearchCV(SVC(), param_grid)
elif args["classifier"]=='knn':
    param_grid = {'n_neighbors': [1,3,5,7,9,11],
              'weights': ['uniform','distance'],
               'algorithm':['auto','ball_tree','kd_tree','brute'],
             'p':[1,2]}
    model=GridSearchCV(KNeighborsClassifier(), param_grid)
    
elif args["classifier"]=='lda':
    param_grid = {'solver':['svd','lsqr','eigen'],
    
    }
    model=GridSearchCV(LinearDiscriminantAnalysis(), param_grid)
    
else:
    print('input correct classifier!')
    exit(-1)
# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model=model.fit(trainX, trainY)

print('Best estimator found by grid search : ')
print(model.best_estimator_)
# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print('{} prediction :'.format(args["classifier"]))
print(classification_report(testY, predictions))

