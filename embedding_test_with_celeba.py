# USAGE
# python find_estimator.py --input caltech_faces -m svm

# import the necessary packages
from sklearn.neural_network import MLPClassifier
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
import load_embedding
import matplotlib.pyplot as plt
# input argument
#required args : face detector dir, embedding model dir, dataset dir, face detaction confident
#Usage : python find_best_estimator_with_embedding.py -i C:/Users/sorjt/Desktop/drive-download-20210429T145212Z-001/archive/img_align_celeba/final
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,help="path to input directory of images")
ap.add_argument("-f", "--face", type=str,default="face_detector",help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=150,help="# of principal components")
ap.add_argument("-d","--detector",type=str,default='face_detector',help="load face detector")
ap.add_argument("-m","--classifier",type=str,default='svm')
ap.add_argument("-e","--embedding_model",type=str,default='./extract_model/openface_nn4.small2.v1.t7',help="load embedding model")
args = vars(ap.parse_args())

# load face detector
# load the image dataset
print("[INFO] loading dataset...")
(faces, labels) = load_embedding.load_data(args)
print("[INFO] {} images in dataset".format(len(faces)))


le = LabelEncoder()
labels = le.fit_transform(labels)

# construct our training and testing split
split = train_test_split(faces,labels, test_size=0.2,stratify=labels, random_state=42)
(trainX, testX, trainY, testY) = split

#model selection
model_can=['SVM','KNN','LDA','MLPC']
precision=[]



for clf in model_can:
    if clf!='MLPC':
        if clf=='SVM':
            param_grid = {
                    'C': [1,10,100,1e3, 5e3, 1e4, 5e4, 1e5],
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],                         
                    'kernel':['linear','poly','rbf','sigmoid'],
                    'degree':[1,2,3,4,5],
                    'gamma':['scale','auto',0.001,0.01,0.1],
                    'class_weight':[None,'balanced']}
            model = GridSearchCV(SVC(), param_grid)
        elif clf=='KNN':
            param_grid = {'n_neighbors': [1,3,5,7,9,11],
                    'weights': ['uniform','distance'],
                        'algorithm':['auto','ball_tree','kd_tree','brute'],
                        'p':[1,2]}
            model=GridSearchCV(KNeighborsClassifier(), param_grid)
                
        elif clf=='LDA':
            param_grid = {'solver':['svd','lsqr','eigen'],
                
                } 
            model=GridSearchCV(LinearDiscriminantAnalysis(), param_grid)
        else:
            print('input correct classifier!')
            exit(-1)
            # train a classifier on the eigenfaces representation
            

        model=model.fit(trainX, trainY)

        print('Best estimator found by grid search : ')
        print(model.best_estimator_)
        #evaluate the model
        print("[INFO] evaluating model...")
        predictions = model.score(testX,testY)
        precision.append(predictions)
    else:
        model=MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)
        model.fit(trainX,trainY)
        precision.append(model.score(testX,testY))

x=np.arange(4)
plt.bar(x,precision)
plt.xticks(x,model_can)
plt.title('feature extracted by Embedding')
plt.grid()
plt.show()

    
