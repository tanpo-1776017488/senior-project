# Real-time mosaic system based on face recognition
이 프로젝트는 종합설계과제로 어떤 어떤겁니다 간략소개
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
