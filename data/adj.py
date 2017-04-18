"""Build adjacency matrices.
"""

import data
from scipy.sparse import dok_matrix


def build_adjacency_matrix(X):
    n = X.shape[0]
    X_adj = dok_matrix((n, n))
    for row in X:
        i, j, w = row
        X_adj[i, j] = w
    return X_adj


if __name__ == '__main__':
    X_train, X_test = data.load()
    X_train_adj = build_adjacency_matrix(X_train)
    X_train_adj = X_train_adj.tocsr()
    data.save_sparse_csr('train_X_adj.npz', X_train_adj)