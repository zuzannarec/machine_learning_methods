import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat  
from scipy import stats
from sklearn import metrics
 
 
data = loadmat('ex8data1.mat')
X = data['X']
print("(amount of data, number of features):", X.shape)

plt.scatter(X[:, 0], X[:, 1])
plt.title("server computers response")
plt.xlabel("throughput (mb/s)")
plt.ylabel("latency (ms).")
plt.show()
plt.hist(X)
plt.show()

def estimate_gaussian(X):
    mu_th = np.mean(X[:, 0])
    mu_l = np.mean(X[:, 1])
    mu = np.array([mu_th, mu_l])
    sigma_th = np.var(X[:, 0])
    sigma_l = np.var(X[:, 1])
    sigma = np.array([sigma_th, sigma_l])
    return mu, sigma

mu_1, sigma_1 = estimate_gaussian(X)
print("mean:", mu_1, "variance:", sigma_1)


Xval = data['Xval'] # features
yval = data['yval'] # labels
print("yval (amount of data, number of features):", yval.shape)
print("Xval (amount of data, number of features):", Xval.shape)

# the probability for X data (first dimension)
dist = stats.norm(mu_1[0], sigma_1[0])
# probability that the first dimension belongs to the distribution
# defined by calculating the mean and variance for  that dimension
# (how far each instance is from the mean)
dist = dist.pdf(X[:,0])
print(dist[0:50])
# p_1 = stats.norm.pdf(X[:,0], mu_1[0], sigma_1[0])
# p_2 = stats.norm.pdf(X[:,1], mu_1[1], sigma_1[1])

# # the probability for Xval data
pval = np.zeros((Xval.shape[0], Xval.shape[1])) # distance for 2 features
pval[:, 0] = stats.norm(mu_1[0], sigma_1[0]).pdf(Xval[:,0])
pval[:, 1] = stats.norm(mu_1[1], sigma_1[1]).pdf(Xval[:,1])
print("pval (amount of data, number of features):", pval.shape)
print("yval (amount of data, number of features):", yval.shape)
def select_threshold(pval, yval):
    p_min = pval.min() # return min (scalar) of the entire array
    p_max = pval.max() # return max (scalar) of the entire array
    step = (p_max - p_min) / 1000
    best_epsilon = p_min
    best_f1 = 0
    f1 = 0
    # prob > threshold - ok (0)
    # prob < threshold - anomaly (1)
    for epsilon in np.arange(p_min, p_max, step): # for each epsilon compare the whole pval array
        preds = pval < epsilon
        true_positive = np.sum(np.logical_and(yval == 1, preds == 1)).astype(np.float) # pred 1, yval 1
        false_positive = np.sum(np.logical_and(yval == 0, preds == 1)).astype(np.float) # pred 1, yval 0
        false_negative = np.sum(np.logical_and(yval == 1, preds == 0)).astype(np.float) # pred 0, yval 1
        recall = true_positive/(true_positive + false_negative)
        precision = true_positive/(true_positive + false_positive)
        f1 = 2 * recall * precision / (recall + precision)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

best_epsilon, best_f1 = select_threshold(pval, yval)
print("best epsilon:", best_epsilon, "best_f1:", best_f1)

outliers = np.where(pval < best_epsilon)
print(outliers)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[outliers[0], 0], X[outliers[0], 1], color='r')
plt.title("server computers response")
plt.xlabel("throughput (mb/s)")
plt.ylabel("latency (ms).")
plt.show()