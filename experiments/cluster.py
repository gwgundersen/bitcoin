"""Clustering and visualization.
"""

import data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


N_TRANSACTIONS = 3348026
N_ADDRESSES = 444075

N_CLUSTERS = 7
COLORS = ['b', 'r', 'k', 'g', 'c', 'm', 'y']


def scatterplot(X):
    x = X[:, 0]
    y = X[:, 2]
    plt.scatter(x, y, color='k', s=3)
    plt.show()


def plot_degrees():
    X_train, X_test = data.load(ds_type='degree')
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    fitted = kmeans.fit(X_train[:,1:])  # Drop addresses.
    clusters = []
    for i in range(N_CLUSTERS):
        indices = fitted.labels_ == i
        clusters.append(X_train[indices])
    scatterplot(clusters)


def plot_raw():
    X_train, _ = data.load()
    np.random.shuffle(X_train)
    scatterplot(X_train)


def dumb_script(X_arg):
    X = X_arg.copy()
    addrs = set()
    for _ in range(1000):
        ind = np.argmax(X[:, 2])
        address_w_max = X[ind][0]
        addrs.add(address_w_max)
        X = np.delete(X, ind, axis=0)
    with open('Addresses with max # transactions.txt', 'w+') as f:
        f.write(str(addrs))
        f.write(str(max(addrs)))


if __name__ == '__main__':
    #X_train, _ = data.load()
    #dumb_script(X_train)
    plot_raw()