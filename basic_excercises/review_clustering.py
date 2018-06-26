import pandas as pd
import numpy as np
from sklearn import datasets, cluster
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data,
                  columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
class_names = iris.target_names.tolist()
df['Species'] = pd.Categorical([class_names[c_inx] for c_inx in iris.target])
estimator = cluster.KMeans(n_clusters=3, random_state=1)
estimator.fit(df.drop('Species', axis=1))
labels = estimator.labels_

fig = plt.figure(1, figsize=(6, 4))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

colors = ['red', 'green', 'blue']
categories = ['Versicolor', 'Setosa', 'Virginica']
for i in np.unique(labels.astype(np.int)):
    category_data = df[labels == i]
    name = categories[i]
    ax.scatter(xs=category_data.PetalWidth, ys=category_data.SepalLength,
               zs=category_data.PetalWidth, c=colors[i],
               label=name, edgecolor='k')
    ax.text3D(category_data.PetalWidth.mean(),
              category_data.SepalLength.mean(),
              category_data.PetalLength.mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
ax.set_xlabel('PetalWidth')
ax.set_ylabel('SepalLength')
ax.set_zlabel('PetalLength')
ax.set_title('K-means')
plt.legend(), plt.show()
sns.lmplot(x="SepalLength", y="SepalWidth", data=df, fit_reg=False, hue='Species', legend=False)
plt.legend(loc='lower right')
plt.show()
index = pd.DataFrame({'cluster': pd.Categorical([categories[c_inx] for c_inx in labels])})
sns.pairplot(pd.concat([df, index], axis=1).drop('Species', axis=1), hue='cluster')
plt.show()
