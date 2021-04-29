from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

def classifier_test(clf,x_test,x_train,y_test,y_train,md,arr):
    clf.fit(x_train,y_train)
    accuracy_train= clf.score(x_train, y_train)
    accuracy_test= clf.score(x_test, y_test)
    arr.append(accuracy_test)
    print('{} classifier Accuracy with train data: {}'.format(md,accuracy_train))
    print('{} classifier Accuracy with test data : {}'.format(md,accuracy_test))

def EV(ran,acc,name):
    #plot
    plt.plot(range(50,ran+1,50), acc)
    plt.title('dimension affect to accuracy({})'.format(name),fontsize=15)
    plt.xlabel('Number of Principle Components ')
    plt.ylabel('accuracy')

def time_EV(ran,acc,name):
    plt.plot(range(50,ran+1,50), acc)
    plt.title('dimension affect to time({})'.format(name),fontsize=15)
    plt.xlabel('Number of Principle Components ')
    plt.ylabel('time')

def EV_cor(n,x_train,x_test):
    pca=PCA(n_components=n,whiten=True)
    X_train=pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    plt.plot(range(1,n+1), pca.explained_variance_ratio_.cumsum())
    plt.title('Explained Variance',fontsize=15)
    plt.xlabel('Number of Principle Components : {}'.format(n), fontsize=10)
    plt.grid(True)

if __name__=="__main__":
    #prepare data
    data=fetch_lfw_people(min_faces_per_person=50)
    
    inputs=data.data
    target=data.target
    images=data.images
    n_sample=images.shape[0]
    n_feature=inputs.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(inputs, target, random_state=365)
    
    # StandardScaler().fit(x_train)
    # StandardScaler().fit(x_test)
    # plt.figure(figsize=(15,5))
    # plt.subplot(2,3,1)
    # EV_cor(50,x_train,x_test)
    # plt.subplot(2,3,2)
    # EV_cor(100,x_train,x_test)
    # plt.subplot(2,3,3)
    # EV_cor(150,x_train,x_test)
    # plt.subplot(2,3,4)
    # EV_cor(200,x_train,x_test)
    # plt.subplot(2,3,5)
    # EV_cor(250,x_train,x_test)
    # plt.subplot(2,3,6)
    # EV_cor(300,x_train,x_test)
    # plt.tight_layout()
    
    # plt.show()

    print('iteration : ',min(n_sample,n_feature))
    knn_acc=[]
    lda_acc=[]
    svm_acc=[]
    knn_t=[]
    lda_t=[]
    svm_t=[]
    for component in range(50,301,50):
        
        pca = PCA(n_components=component, whiten=True)
        X_train = pca.fit_transform(x_train)
        X_test = pca.transform(x_test)
        print('dimension : ',component)
        print('\nsvm : ')
        svm_clf=svm.SVC(kernel="rbf",C=10,gamma=0.001,random_state=42)
        before=time.time()
        classifier_test(svm_clf,X_test,X_train,y_test,y_train,'svm',svm_acc)
        after=time.time()
        print('elapsed time : ',after-before)
        svm_t.append(after-before)

        print('\nknn : ')
        knn_clf=KNeighborsClassifier(n_neighbors=3,weights="distance")
        before=time.time()
        classifier_test(knn_clf,X_test,X_train,y_test,y_train,'knn',knn_acc)
        after=time.time()
        print('elapsed time : ',after-before)
        knn_t.append(after-before)
        print('\n')

        print('LDA : ')
        LDA_clf=LinearDiscriminantAnalysis()
        before=time.time()
        classifier_test(LDA_clf,X_test,X_train,y_test,y_train,'LDA',lda_acc)
        after=time.time()
        print('elapsed time : ',after-before)
        lda_t.append(after-before)

        # print('\n MLPC : ')
        # MLPC_clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)
        # before=time.time()
        # classifier_test(MLPC_clf,X_test,X_train,y_test,y_train,'MLPC')
        # after=time.time()
        # print('elapsed time : ',after-before)
    plt.figure(figsize=(15,5))
    plt.subplot(2,3,1)
    EV(300,svm_acc,'SVM')

    plt.subplot(2,3,2)
    EV(300,knn_acc,'KNN')

    plt.subplot(2,3,3)
    EV(300,lda_acc,'LDA')

    plt.subplot(2,3,4)
    time_EV(300,svm_t,'SVM')

    plt.subplot(2,3,5)
    time_EV(300,knn_t,'KNN')

    plt.subplot(2,3,6)
    time_EV(300,lda_t,'LDA')
    plt.tight_layout()
    plt.show()


    
    
    
    
    
