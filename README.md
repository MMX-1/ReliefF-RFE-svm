# ReliefF-RFE-svm
We propose a novel sparse SVM, named as ReliefF based on SVM, which combines recursive feature elimination (RFE) and ReliefF using a weight parameter. This new filter algorithm can capture relevant features and feature interactgions simultaneously and is crucial in preventing valuable features from being removed at each iteration.
# linear ReliefF-RFE-SVM
## $\alpha$ and optimal feature selection
First, obtaining optimal feature number and $\alpha$ by using average accuracy. For example, we take as $\alpha=\{0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95\}$.

### Input Parameter

-alpha: weight parameter $\alpha$, $0\le \alpha\le 1$ controls the trade-off between ReliefF ranking and SVMs-RFE ranking.

-folds: folds of cross-validation 

-optim_featur_num: optimal feature number of feature subset

```code
rfe_relief_model=rfe_relief_SVM(x_train,y_train,0.1)
aver_mean_lis1,F_measure_lis1=rfe_relief_model.featu_score(0.15,5)
print(aver_mean_lis1,F_measure_lis1)

aver_mean_lis2,F_measure_lis2=rfe_relief_model.featu_score(0.25,5)
print(aver_mean_lis2,F_measure_lis2)

aver_mean_lis3,F_measure_lis3=rfe_relief_model.featu_score(0.35,5)
print(aver_mean_lis3,F_measure_lis3)

aver_mean_lis4,F_measure_lis4=rfe_relief_model.featu_score(0.45,5)
print(aver_mean_lis4,F_measure_lis4)

aver_mean_lis5,F_measure_lis5=rfe_relief_model.featu_score(0.55,5)
print(aver_mean_lis5,F_measure_lis5)


aver_mean_lis6,F_measure_lis6=rfe_relief_model.featu_score(0.65,5)
print(aver_mean_lis6,F_measure_lis6)

aver_mean_lis7,F_measure_lis7=rfe_relief_model.featu_score(0.75,5)
print(aver_mean_lis7,F_measure_lis7)

aver_mean_lis8,F_measure_lis8=rfe_relief_model.featu_score(0.85,5)
print(aver_mean_lis8,F_measure_lis8)

aver_mean_lis9,F_measure_lis9=rfe_relief_model.featu_score(0.95,5)
print(aver_mean_lis9,F_measure_lis9)
```



Next, identify optimal feature number and $\alpha$ by finding the highest average highest of the first $k_{max}$ average accuracy from nine average accuracy list, for example $k_{max}=13$

```code
max(aver_mean_lis1[:13]),np.max(aver_mean_lis2[:13]),np.max(aver_mean_lis3[:13]),np.max(aver_mean_lis4[:13]),np.max(aver_mean_lis5[:13]),np.max(aver_mean_lis6[:13]),np.max(aver_mean_lis7[:13]),np.max(aver_mean_lis8[:13]),np.max(aver_mean_lis9[:13])
```
Then we can observe that optimal $\alpha$ corresponding to the higheset average accuracy, for example $\alpha=0.35$.

```code
np.argmax(aver_mean_lis3[:13])
```
obtain optimal feature number, for example, optimal feature number is 11.


## obtain optimal technical indicators and performance of SVC
```code

rfe_relief_SVM_model=rfe_relief_SVM(x_train,y_train,0.1)
rfe_relief_SVM_sele_featu_subset=rfe_relief_SVM_model.sele_feature(0.35,11)
#obtain optimal technical indicators and feautre number
np.array(techini_indicator)[rfe_relief_SVM_sele_featu_subset],len(np.array(techini_indicator)[rfe_relief_SVM_sele_featu_subset])

#calculate accuracy,recall,specificity,precision,F-test
estimator_SVM_rfe_reliefF=SVC(kernel='linear',C=0.1)
estimator_SVM_rfe_reliefF.fit(x_train[:,rfe_relief_SVM_sele_featu_subset],y_train)
rfe_reliefF_SVM_result=estimator_SVM_rfe_reliefF.predict(x_test[:,rfe_relief_SVM_sele_featu_subset])
performan_rfe_reliefF_SVM=meansure_performance(rfe_reliefF_SVM_result)
print(performan_rfe_reliefF_SVM)

```

# Gaussian kernel SVM-RFE

## $\alpha$ and optimal feature selection
First, obtaining optimal feature number and $\alpha$ by using average accuracy. For example, we take as $\alpha=\{0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95\}$.

### Input Parameter

-alpha: weight parameter $\alpha$, $0\le \alpha\le 1$ controls the trade-off between ReliefF ranking and SVMs-RFE ranking.

-folds: folds of cross-validation 

-optim_featur_num: optimal feature number of feature subset

-gamma: parameter of gaussian kernel function

```code
rfe_relief_model=rfe_relief_SVM(x_train,y_train,2,0.1)
aver_mean_lis1,F_measure_lis1=rfe_relief_model.featu_score(0.15,5)
print(aver_mean_lis1,F_measure_lis1)

aver_mean_lis2,F_measure_lis2=rfe_relief_model.featu_score(0.25,5)
print(aver_mean_lis2,F_measure_lis2)


aver_mean_lis3,F_measure_lis3=rfe_relief_model.featu_score(0.35,5)
print(aver_mean_lis3,F_measure_lis3)

aver_mean_lis4,F_measure_lis4=rfe_relief_model.featu_score(0.45,5)
print(aver_mean_lis4,F_measure_lis4)

aver_mean_lis5,F_measure_lis5=rfe_relief_model.featu_score(0.55,5)
print(aver_mean_lis5,F_measure_lis5)

aver_mean_lis6,F_measure_lis6=rfe_relief_model.featu_score(0.65,5)
print(aver_mean_lis6,F_measure_lis6)

aver_mean_lis7,F_measure_lis7=rfe_relief_model.featu_score(0.75,5)
print(aver_mean_lis7,F_measure_lis7)

aver_mean_lis8,F_measure_lis8=rfe_relief_model.featu_score(0.85,5)
print(aver_mean_lis8,F_measure_lis8)

aver_mean_lis9,F_measure_lis9=rfe_relief_model.featu_score(0.95,5)
print(aver_mean_lis9,F_measure_lis9)
```
Next, identify optimal feature number and $\alpha$ by finding the highest average highest of the first $k_{max}$ average accuracy from nine average accuracy list, for example $k_{max}=13$

```code
np.max(aver_mean_lis1[:9]),np.max(aver_mean_lis2[:9]),np.max(aver_mean_lis3[:9]),np.max(aver_mean_lis4[:9]),np.max(aver_mean_lis5[:9]),np.max(aver_mean_lis6[:9]),np.max(aver_mean_lis7[:9]),np.max(aver_mean_lis8[:9]),np.max(aver_mean_lis9[:9])
```

Then we can observe that optimal $\alpha$ corresponding to the higheset average accuracy, for example $\alpha=0.95$.

```code
np.argmax(aver_mean_lis9[:9])
```
obtain optimal feature number, for example, optimal feature number is 9.
## obtain optimal technical indicators and performance of SVC

```code
rfe_relief_SVM_model=rfe_relief_SVM(x_train,y_train,1.5,0.1)
rfe_relief_SVM_sele_featu_subset=rfe_relief_SVM_model.sele_feature(0.95,7)
#obtain optimal technical indicators and feautre number
np.array(techini_indicator)[rfe_relief_SVM_sele_featu_subset],len(np.array(techini_indicator)[rfe_relief_SVM_sele_featu_subset])
#calculate accuracy,recall,specificity,precision,F-test
estimator_SVM_rfe_reliefF=SVC(C=1.5, kernel='rbf',gamma=0.1)
estimator_SVM_rfe_reliefF.fit(x_train[:,rfe_relief_SVM_sele_featu_subset],y_train)
rfe_reliefF_SVM_result=estimator_SVM_rfe_reliefF.predict(x_test[:,rfe_relief_SVM_sele_featu_subset])
performan_rfe_reliefF_SVM=meansure_performance(rfe_reliefF_SVM_result)
print(performan_rfe_reliefF_SVM)
```
# Reference
Miao, Maoxuan, Jinran Wu, Fengjing Cai, Liya Fu, Shurong Zheng, and You‐Gan Wang. "Feature Selection for Stock Movement Direction Prediction using Sparse Support Vector Machine."
