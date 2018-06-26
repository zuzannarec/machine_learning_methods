from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

wine = load_wine()

clf_tree = DecisionTreeClassifier(random_state=1, min_samples_leaf=3)
scores_tree = cross_val_score(clf_tree, wine.data, wine.target, cv=5)
print("Decision Tree Classifier:", np.around(scores_tree, decimals=2),
      "(avg)", np.around(np.mean(scores_tree), decimals=3))


clf_bag = BaggingClassifier(random_state=1, n_estimators=50)
scores_bag = cross_val_score(clf_bag, wine.data, wine.target, cv=5)
print("Bagging Classifier:", np.around(scores_bag, decimals=2),
      "(avg)", np.around(np.mean(scores_bag), decimals=3))

clf_ada = AdaBoostClassifier(random_state=1, n_estimators=50, algorithm='SAMME')
scores_ada = cross_val_score(clf_ada, wine.data, wine.target, cv=5)
print("Ada Boost Classifier:", np.around(scores_ada, decimals=2),
      "(avg)", np.around(np.mean(scores_ada), decimals=3))

clf_grad = GradientBoostingClassifier(random_state=1, learning_rate=1.0, subsample=0.5, n_estimators=50,
                                      min_samples_leaf=3, max_depth=1)
scores_grad = cross_val_score(clf_grad, wine.data, wine.target, cv=5)
print("Gradient Boosting Classifier:", np.around(scores_grad, decimals=2),
      "(avg)", np.around(np.mean(scores_grad), decimals=3))
clf_grad = clf_grad.fit(wine.data, wine.target)
cumsum = np.cumsum(clf_grad.oob_improvement_)
x = np.arange(50) + 1
plt.scatter(x, cumsum)
plt.xlabel('iteration')
plt.ylabel('improvement')
plt.show()