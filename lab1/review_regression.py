from sklearn import datasets, model_selection, linear_model
import sklearn.metrics
import numpy as np
import pandas as pd
import seaborn
import matplotlib as plt

diabetes = datasets.load_diabetes()
data = np.hstack((diabetes.data, np.array([diabetes.target]).T))
column_names = diabetes.feature_names+['type']
df = pd.DataFrame(data=data,
                  columns=column_names)
train_set, test_set = model_selection.train_test_split(df, test_size=0.3, random_state=1)
X_train = train_set.drop(column_names[-1], axis=1)
X_test = test_set.drop(column_names[-1], axis=1)
Y_train = train_set.drop(column_names[:-1], axis=1)
Y_test = test_set.drop(column_names[:-1], axis=1)
#fit classifierp
regr = linear_model.LinearRegression()
regr.fit(X_train.bmi.values.reshape(-1, 1), Y_train)
prediction = regr.predict(X_test.bmi.values.reshape(-1, 1))
MSE = sklearn.metrics.mean_squared_error(Y_test, prediction)
R2 = sklearn.metrics.r2_score(Y_test, prediction)
coefficient = regr.coef_
intercept = regr.intercept_
print("MSE:", MSE)
print("R2:", R2)
print("coefficient:", coefficient[0][0])
print("intercept:", intercept[0])
seaborn.regplot(x="bmi", y="type", data=test_set)
plt.pyplot.show()
