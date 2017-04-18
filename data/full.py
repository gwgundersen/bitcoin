"""Add degree features to full transaction matrices.
"""

import data
import numpy as np


def add_new_features(X, degree_map):
    new_features = []
    for row in X:
        sender, receiver, n_trans = row
        sender_out, sender_in = degree_map[sender]
        receiver_out, receiver_in = degree_map[receiver]
        new_row = [sender_out, sender_in, receiver_out, receiver_in]
        new_features.append(new_row)
    return np.column_stack((X, new_features))


if __name__ == '__main__':
    X_train, X_test = data.load()
    degree_map = data.load_degree_maps()

    X_train = add_new_features(X_train, degree_map)
    X_test = add_new_features(X_test, degree_map)

    np.save('data/train_X_full.npy', X_train)
    np.save('data/test_X_full.npy', X_test)
