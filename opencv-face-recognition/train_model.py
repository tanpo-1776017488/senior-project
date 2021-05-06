# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import neighbors
import argparse
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

scaler=StandardScaler()
pca=PCA(n_components=100,whiten=True)
data["embeddings"]=scaler.fit_transform(data["embeddings"])
data["embeddings"]=pca.fit_transform(data["embeddings"])
print(data["embeddings"])
print("[INFO] training model...")
recognizer = SVC(gamma='scale', probability=True)
#recognizer=neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()