"""Return subsample of full dataset.
"""

import data
from data.full import add_new_features
import numpy as np


N_ADDRESSES = 444075


def add_degree(X, degree):
    idx = 0 if degree == 'out' else 1
    nodes, inv, counts = np.unique(X[:, idx], return_inverse=True,
                                   return_counts=True)
    return np.column_stack((X, counts[inv]))


def add_degree_cols(X):
    X = add_degree(X, 'out')
    X = add_degree(X, 'in')
    return X


def subsample(n, pct_pos):
    X_train, X_test = data.load('full')
    degree_map = data.load_degree_maps()

    pos_map = pos_transaction_hash(X_train)
    added = {}

    np.random.shuffle(X_train)
    n_pos = round(n * pct_pos)
    n_neg = n - n_pos
    X_train_pos = X_train[:n_pos]
    X_train_neg = []
    while len(X_train_neg) < n_neg:
        i = np.random.randint(0, N_ADDRESSES)
        j = np.random.randint(0, N_ADDRESSES)
        if (i, j) in pos_map or (i, j) in added:
            continue
        else:
            added[i, j] = True
            X_train_neg.append([i, j, 0])

    X_train_neg = np.array(X_train_neg)
    X_train_neg = add_new_features(X_train_neg, degree_map)
    assert X_train_pos.shape[0] + X_train_neg.shape[0] == n

    X_train = np.row_stack((X_train_pos, X_train_neg))
    np.random.shuffle(X_train)  # Mix labels together.
    return X_train, X_test


def pos_transaction_hash(X_train):
    pos = {}
    for row in X_train:
        i, j, _, _, _, _, _ = row
        pos[i, j] = True
    return pos


def create_labels(X_train, X_test):
    y_train = X_train[:, 2]
    y_train = (y_train != 0).astype(int)
    X_train = np.delete(X_train, 2, axis=1)
    y_test  = X_test[:, 2]
    y_test  = (y_test != 0).astype(int)
    X_test = np.delete(X_test, 2, axis=1)
    return X_train, y_train, X_test, y_test


def load(n, split):
    X_train, X_test = subsample(n, split)
    return create_labels(X_train, X_test)
