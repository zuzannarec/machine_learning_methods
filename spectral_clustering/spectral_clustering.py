# -*- coding: utf-8 -*-
"""

    TOPIC: Spectral clustering

"""

import numpy as np
from sklearn import cluster, datasets
import networkx as nx

import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph


RANDOM_STATE = 1

#%% Auxiliary functions used to generate datasets.

def generate_k2_circles():
    X, _ = datasets.make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=25)
    return X

def generate_k2_moons():
    X, _ = datasets.make_moons(n_samples=400, noise=.05)
    return X
    
def generate_k3():
    moons = datasets.make_moons(n_samples=200, noise=.1, random_state=10)
    Xm, ym = moons
    Xm1 = Xm[ym == 0]

    T = np.array([[0, 0.1]])
    Xm1 = Xm1 + T  # translation
    
    moons = datasets.make_moons(n_samples=400, noise=.1, random_state=10)
    Xm, ym = moons
    Xm2 = Xm[ym == 1]
    
    blobs = datasets.make_blobs(n_samples=100, centers=1, center_box=(0, 0), cluster_std=0.5,
                                random_state=8)
    Xm, ym = blobs
    Xm3 = Xm
    
    S = [[1.0, 0], [0, 0.5]]
    Xm3 = np.dot(Xm3, S)  # scaling
    
    alpha = np.radians(15)
    R = [
         [np.cos(alpha), -np.sin(alpha)],
         [np.sin(alpha), np.cos(alpha)]
        ]
    Xm3 = np.dot(Xm3, R)  # rotation
    
    T = np.array([[-0.5, -1.1]])
    Xm3 = Xm3 + T  # translation
    
    X = np.concatenate((Xm1, Xm2, Xm3), axis=0)
    return X


#%% Generate the dataset.

N_CLUSTERS = 3  # Choose between {2, 3}

# TODO: Observe differences for N = {10, 15, 20}
N_NEIGHBORS = 20

if N_CLUSTERS == 2:
    # Two modes available
    X = generate_k2_circles()
#    X = generate_k2_moons()
elif N_CLUSTERS == 3:
    X = generate_k3()
    
eigen_solver = 'arpack'
    
#%% Visualize the dataset.

_, ax = plt.subplots(1,1, num=1)
ax.set_title("Dataset")
ax.plot(X[:,0], X[:,1], 'go')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
ax.set_aspect('equal')

#%% Set clustering models.

model_kmeans = cluster.KMeans(n_clusters=N_CLUSTERS, random_state=1)
model_spectralnn = cluster.SpectralClustering(n_clusters=N_CLUSTERS,
                                              affinity='nearest_neighbors',
                                              n_neighbors=N_NEIGHBORS,
                                              random_state=1)

#%% Fit clustering models (k-means and spectral clustering using "ordinary" k-NN).

model_kmeans = model_kmeans.fit(X)
model_spectralnn = model_spectralnn.fit(X)
print(model_kmeans)
print(model_spectralnn)
#%% Visualize clustering results (k-means vs SC using "ordinary" k-NN).
def plot_clustering(ax, X, title, model):
    markers = ['.', '^', '*']
    n_clusters = model.n_clusters
    for i in range(n_clusters):
        is_cluster_observation = (model.labels_ == i)
        ax.plot(X[is_cluster_observation, 0], X[is_cluster_observation, 1],
                markers[i])
        # --
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)
        ax.set_aspect('equal')


fig, (ax_kmeans, ax_spect) = plt.subplots(1, 2, num=2, subplot_kw={'aspect': 1.0})
plt.subplots_adjust(wspace=0.5)
plot_clustering(ax_kmeans, X, "K-means", model_kmeans)
plot_clustering(ax_spect, X, "Spectral clustering", model_spectralnn)
plt.show()

#%% Show connections in graphs (visualize the affinity matrix).

FIG_SIZE = 7
NODE_SIZE = 50
_, (ax_norm, ax_mut) = plt.subplots(1,2, num=4, figsize=(2 * FIG_SIZE, FIG_SIZE))

W = model_spectralnn.affinity_matrix_
G = nx.from_scipy_sparse_matrix(W)
nx.draw(G, pos=X, ax=ax_norm, node_color='g', node_size=NODE_SIZE)
ax_norm.set_aspect('equal')
ax_norm.set_title('Regular kNN')

W0 = kneighbors_graph(X, N_NEIGHBORS, mode='connectivity', include_self=True).toarray()
W0T = W0.transpose()
W_mut = W0 * W0T
G_mut = nx.from_numpy_matrix(W_mut)
nx.draw(G_mut, pos=X, ax=ax_mut, node_color='g', node_size=NODE_SIZE)
ax_mut.set_aspect('equal')
ax_mut.set_title('Mutual kNN')

#%% Fit clustering model (spectral clustering using mutual "ordinary" k-NN)

model_spectralnn_mut = cluster.SpectralClustering(n_clusters=N_CLUSTERS,
                                                  eigen_solver=eigen_solver,
                                                  affinity='precomputed',
                                                  random_state=RANDOM_STATE)
model_spectralnn_mut.fit(W_mut)

#%% Visualize clustering results (SC using "ordinary" k-NN vs using mutual k-NN).

fig, (ax_norm, ax_mutual) = plt.subplots(1,2, num=5, subplot_kw={'aspect': 1.0})
plt.subplots_adjust(wspace = 0.5)
plot_clustering(ax_norm, X, "Normal kNN", model_spectralnn)
plot_clustering(ax_mutual, X, "Mutual kNN", model_spectralnn_mut)
plt.show()
