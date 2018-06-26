from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


dataset = load_boston()
df = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']],
                  columns=np.append(dataset['feature_names'], ['TARGET']))
x = df.filter(items=dataset.feature_names)
y = df['TARGET']
for index, feature_name in enumerate(dataset.feature_names):
    plt.scatter(x[feature_name], y)
    plt.xlabel("x - " + feature_name)
    plt.ylabel("y - MEDV (target")
    plt.title("Feature {}\n".format(feature_name))
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

degrees = [2, 5]
for i in degrees:
    polynomial_features = PolynomialFeatures(degree=i,
                                             include_bias=False)
    # Create linear regression object
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X_train, y_train)
    # Evaluate the models using crossvalidation
    scoresMSE = cross_val_score(pipeline, x, y,
                                scoring="neg_mean_squared_error", cv=10)
    scoresr2 = cross_val_score(pipeline, x, y,
                               scoring="r2", cv=10)
    print("Degree {}\nMSE = {:.2e}(+/- {:.2e})\nr2 = {:.2e}(+/- {:.2e})".format(
          i, -scoresMSE.mean(), scoresMSE.std(), -scoresr2.mean(), scoresr2.std()))
    print("Coefficients:", linear_regression.coef_)

# --- Ridge regression ---
# Linear Model with Ridge
ridge = linear_model.Ridge(alpha=0.5)
# Model fitting
ridge.fit(X_train, y_train)
# Prediction
prediction = ridge.predict(X_test)
# Compute MSE, R2
MSE = metrics.mean_squared_error(y_test, prediction)
R2 = metrics.r2_score(y_test, prediction)
print("MSE:", MSE)
print("R2:", R2)

# --- Lasso ---
lasso = linear_model.Lasso(alpha=0.5)
# Model fitting
lasso.fit(X_train, y_train)
# Prediction
prediction_lasso = lasso.predict(X_test)
# Compute MSE, R2
MSE = metrics.mean_squared_error(y_test, prediction_lasso)
R2 = metrics.r2_score(y_test, prediction_lasso)
print("MSE Lasso:", MSE)
print("R2 Lasso:", R2)

# --- Regularization params ---
alpha_array = 10**np.linspace(10,-2,100)*0.5
ridge_reg = linear_model.Ridge()
lasso_reg = linear_model.Lasso()
cv_ridge = list()
cv_lasso = list()
coefficients_ridge = list()
coefficients_lasso = list()
for i in alpha_array:
    ridge_reg.set_params(alpha=i)
    ridge_reg.fit(x, y)
    lasso_reg.set_params(alpha=i)
    lasso_reg.fit(x, y)
    cv_ridge.append(cross_val_score(ridge, x, y, cv=3).mean())
    cv_lasso.append(cross_val_score(lasso, x, y, cv=3).mean())
    coefficients_ridge.append(ridge_reg.coef_.mean())
    coefficients_lasso.append(lasso_reg.coef_.mean())

blue_patch = mpatches.Patch(color='b', label='Ridge')
red_patch = mpatches.Patch(color='r', label='Lasso')
# plot a function of mean cross-validation score vs alpha for lasso and ridge regression
plt.scatter(alpha_array, cv_ridge, c='b')
plt.scatter(alpha_array, cv_lasso, c='r')
plt.xlabel("alpha")
plt.ylabel("mean cross-validation score")
plt.title("Mean cross-validation score for ridge regression and lasso regression")
plt.legend(handles=[blue_patch, red_patch])
plt.show()
# plot coefficients associated with each alpha value for ridge and lasso regression
np.shape(coefficients_ridge)
np.shape(coefficients_lasso)
plt.scatter(alpha_array, coefficients_ridge, c='b')
plt.scatter(alpha_array, coefficients_lasso, c='r')
plt.xscale('log')
plt.xlabel("alpha")
plt.ylabel("Coefficients")
plt.title("Coefficients for ridge regression and lasso regression")
plt.legend(handles=[blue_patch, red_patch])
plt.show()

ridgecv = linear_model.RidgeCV(alphas=alpha_array, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
print("Ridge alpha", ridgecv.alpha_)


