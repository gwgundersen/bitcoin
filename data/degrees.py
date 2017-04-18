"""Compute in- and out-degree for every address (node) and save it in a pickled
dictionary.
"""

import data
import numpy as np
import pickle


N_ADDRESSES = 444075


def add(X, M, idx):
    unique, counts = np.unique(X[:, idx], return_counts=True)
    for addr, count in zip(unique, counts):
        M[addr][idx] = count
    return M


if __name__ == '__main__':
    X_train, _ = data.load()
    M = {i: [0, 0] for i in range(N_ADDRESSES)}
    M = add(X_train, M, 0)  # Out degree
    M = add(X_train, M, 1)  # In degree
    with open('data/degree_map.pickle', 'wb') as handle:
        pickle.dump(M, handle, protocol=pickle.HIGHEST_PROTOCOL)
