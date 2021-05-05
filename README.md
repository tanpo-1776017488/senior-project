# Real-time mosaic system based on face recognition
길거리에서 방송할 때 방송에 자신의 얼굴이 나오는지도 모르는채로 초상권 침해가 빈번히 이루어 지고 있음.         
이 현상을 방지하고자 만든 시스템.
## Classifier selection
> * LDA
> * MLPC
> * SVM
> * KNN

## Feature extraction
> * PCA(Principal Component Analysis)
> * Embedding


## Dataset
 ```from sklearn.datasets import fetch_lfw_people```
> * Experiment (using PCA)
![Alt text](Fig/skleran_dataset_output/component_changes_variance.png)
![Alt text](Fig/skleran_dataset_output/component_changes_variance_mlpc_added.png)

## Future plans
> 1. PCA + Classifier 조합을 celeba dataset으로 테스트 후 classifier결정.
> 2. PCA + Classifier 조합과 Embedding + Classifier 조합 성능비교.

## Reference
* [celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
