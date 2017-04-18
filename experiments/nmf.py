"""Generate a covariance matrix.
"""

import data
import numpy as np
from sklearn.decomposition import NMF


def predict(W, H, X_test, threshold=1e-8):
    preds = np.zeros(X_test.shape[0])
    c = 0
    for i,j,w in X_test:
        w = W[i, :]
        h = H[:, j]
        p = np.dot(w, h)
        preds[c] = 1 if p > threshold else 0
    return preds


if __name__ == '__main__':
    X_train, X_test = data.load('adjacency')
    print('Data loaded.')
    model = NMF(n_components=10)
    W = model.fit_transform(X_train)
    H = model.components_
    preds = predict(W, H, X_test)
    import pdb; pdb.set_trace()

