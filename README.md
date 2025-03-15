# ReliefF-RFE-svm
We propose a novel sparse SVM, named as ReliefF based on SVM, which combines recursive feature elimination (RFE) and ReliefF using a weight parameter. This new filter algorithm can capture relevant features and feature interactgions simultaneously and is crucial in preventing valuable features from being removed at each iteration.
# linear ReliefF-RFE-SVM
## $\alpha$ and optimal feature selection
First, obtaining optimal feature number and $\alpha$ by using average accuracy. For example, we take as $\alpha=\{0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95\}$.

## Paramet

-alpha: weight parameter $\alpha$, $0\le \alpha\le 1$ controls the trade-off between ReliefF ranking and SVMs-RFE ranking.

-folds: folds of cross-validation 

-optim_featur_num: optimal feature number of feature subset

```python
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
'''
---



Next, identify optimal feature number and $\alpha$ by finding the highest average highest of the first $k_{max}$ average accuracy from nine average accuracy list, for example $k_{max}=13$
'''python
max(aver_mean_lis1[:13]),np.max(aver_mean_lis2[:13]),np.max(aver_mean_lis3[:13]),np.max(aver_mean_lis4[:13]),np.max(aver_mean_lis5[:13]),np.max(aver_mean_lis6[:13]),np.max(aver_mean_lis7[:13]),np.max(aver_mean_lis8[:13]),np.max(aver_mean_lis9[:13])
'''
