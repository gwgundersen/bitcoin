"""Generate two new features, one for in-degree and one for out-degree of each
node.
"""

import data
import numpy as np


N_ADDRESSES = 444075


def gen_degree_columns(X, out_degree):
    """Generate column with counts of in- or out-degrees. Ensure ordered by
    addresses (0 - 444074).
    """
    X_new = np.zeros((N_ADDRESSES,), dtype=np.int)
    idx = 0 if out_degree else 1
    addresses, degrees = np.unique(X[:, idx], return_counts=True)
    X_new[addresses] = degrees
    # Sanity check that the number of addresses that do not have an in- or out-
    # degree node are equal to the number of zeros in the new data structure.
    assert N_ADDRESSES - addresses.size == (X_new == 0).sum()
    return X_new


def gen_degree_columns_and_save(X, train):
    """Generate data matrix with columns (address, out-degree, in-degree).
    """
    prefix = 'train' if train else 'test'
    fname  = 'data/%s_X_degrees.npy' % prefix
    X_out = gen_degree_columns(X, True)   # Out degrees
    X_in  = gen_degree_columns(X, False)  # In degrees
    addresses = range(0, N_ADDRESSES)
    X_new = np.column_stack((addresses, X_out, X_in))
    np.save(fname, X_new)


if __name__ == '__main__':
    X_train, X_test = data.load()
    gen_degree_columns_and_save(X_train, True)
    gen_degree_columns_and_save(X_test, False)
