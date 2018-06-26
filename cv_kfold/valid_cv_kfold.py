# -*- coding: utf-8 -*-
"""

    TOPIC: K-fold Cross-validation

"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


dataset = datasets.load_iris()

X_all = dataset.data
y = dataset.target

reg = LogisticRegression()
reg = reg.fit(X_all, y)

skf = StratifiedKFold(n_splits=5, random_state=1)

print(np.round(cross_val_score(reg, X_all, y, cv=skf), 2))
# expected: [1.   0.97 0.93 0.9  1.  ]
