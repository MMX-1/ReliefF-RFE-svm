# ReliefF-RFE-svm
We propose a novel sparse SVM, named as ReliefF based on SVM, which combines recursive feature elimination (RFE) and ReliefF using a weight parameter. This new filter algorithm can capture relevant features and feature interactgions simultaneously and is crucial in preventing valuable features from being removed at each iteration.
# linear ReliefF-RFE-SVM
## $\alpha$ and optimal feature selection
First, we calculate average accuracy of any feature subset at each $\alpha$. For example, we take as $\alpha=\{0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95\}$.
```python
def greet(name):
    return f"Hello, {name}!"
