import pandas as pd
import numpy as np
import seaborn
import matplotlib as plt
import sklearn.metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn import tree

iris = datasets.load_iris()
data = np.hstack((iris.data, np.array([iris.target]).T))
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.DataFrame(data=data,
                  columns=column_names)
class_names = iris.target_names.tolist()
df.Species = pd.Categorical([class_names[c_inx] for c_inx in df.Species.astype(int)])
train_set, test_set = model_selection.train_test_split(df, test_size=0.3)
X_train = train_set.drop(column_names[-1], axis=1)
X_test = test_set.drop(column_names[-1], axis=1)
Y_train = train_set.drop(column_names[:-1], axis=1)
Y_test = test_set.drop(column_names[:-1], axis=1)

#fit classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
#use fitted classifier to predict
prediction = clf.predict(X_test)

ct = pd.crosstab(prediction, Y_test.values.T[0],
                 rownames=['Actual'], colnames=['Predicted'], margins=True)
seaborn.heatmap(ct, annot=True, fmt="d")
plt.pyplot.show()

#Calculcate precision and recall
precision = sklearn.metrics.precision_score(Y_test, prediction, average='macro')
recall = sklearn.metrics.recall_score(Y_test, prediction, average='macro')
print("Precision:", precision, "Recall:", recall)
