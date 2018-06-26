import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['target'] = pd.Categorical.from_codes(iris['target'], iris['target_names'])

pl_av = df['petal length (cm)'].mean()
print("number of columns", len(df.columns))
print("average petal length: ", pl_av)
for i in df.columns[:-1]:
    print("average of ", i, "is equal to: ", df[i].mean())
    
pl_outliers_indices = list()
for index, j in enumerate(df['petal length (cm)']):
    if (j > 1.5*pl_av):
        pl_outliers_indices.append(index)
        
print(pl_outliers_indices)

categories = df['target']
print(df.groupby(categories).std())

pl_outliers_indices_per_species = {}
for i in df['target']:
    pl_outliers_indices_per_species[i] = []

for name, group in df['petal length (cm)'].groupby(categories):
    av = 0
    av = np.float64(group.mean())
    for index, j in enumerate(group):
        if (j > 1.5*av):
            pl_outliers_indices_per_species[name].append(index)
print(pl_outliers_indices_per_species.items())

#pd.DataFrame(np.hstack(iris.daya, np.array([iris.target]).T), columns = ['sepal', 'petal', 'name'])
