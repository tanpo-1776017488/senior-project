# USAGE
# python test_performance.py --embeddings output/embeddings.pickle 

# import the necessary packages
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,help="path to serialized db of facial embeddings")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
x_train,x_test,y_train,y_test=train_test_split(data["embeddings"],labels,random_state=42,stratify=labels)
clf=['svm','knn','lda','mlpc']
print("[INFO] training model...")
for clff in clf:
    if clff=='svm':
        param_grid = {'C': [1,10,100,1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
                'kernel':['linear','poly','rbf','sigmoid'],
                'degree':[1,2,3,4,5],
                'gamma':['scale','auto',0.001,0.01,0.1],
                'class_weight':[None,'balanced']}
        recognizer = GridSearchCV(SVC(), param_grid)
    elif clff=='knn':
        param_grid = {'n_neighbors': [1,3,5,7,9,11],
                'weights': ['uniform','distance'],
                'algorithm':['auto','ball_tree','kd_tree','brute'],
                'p':[1,2]}
        recognizer=GridSearchCV(KNeighborsClassifier(), param_grid)
        
    elif clff=='lda':
        param_grid = {'solver':['svd','lsqr','eigen'],
        
        }
        recognizer=GridSearchCV(LinearDiscriminantAnalysis(), param_grid)
    else:
        recognizer=MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)

    recognizer.fit(x_train,y_train)
    if clff!='mlpc':
        print('{} Best estimator found by grid search : ')
        print(recognizer.best_estimator_)
    print('prediction : ',recognizer.score(x_test,y_test))
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - -\n')
#recognizer = SVC(gamma='scale', probability=True)
#recognizer=neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
# recognizer.fit(x_train, y_train)
# print("predict : ",recognizer.score(x_test,y_test))
# print('Best estimator found by grid search : ')
# print(model.best_estimator_)

