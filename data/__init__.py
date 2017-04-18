"""Interface for loading data.
"""

import numpy as np
import pickle
from data import subsample
from scipy.sparse import csr_matrix


DIR = 'data'

DSTYPE_TO_FILES = {
    'default':   ['train_X_raw.npy',  'test_X_raw.npy'],
    'adjacency': ['train_X_adj.npz',  'test_X_raw.npy'],
    'degree':    ['train_X_deg.npy',  'test_X_deg.npy'],
    'full':      ['train_X_full.npy', 'test_X_full.npy'],
    'subsample': []
}


def load(ds_type='default', n=10000, pct_pos=0.5):
    """Load dataset depending on type.
    """
    if ds_type == 'subsample':
        return subsample.load(n, pct_pos)
    train_fname, test_fname = DSTYPE_TO_FILES[ds_type]
    train_path = '%s/%s' % (DIR, train_fname)
    test_path = '%s/%s' % (DIR, test_fname)
    if ds_type == 'adjacency':
        train = load_sparse_csr(train_path)
        test = np.load(test_path)
    else:
        train = np.load(train_path)
        test = np.load(test_path)
    return train, test


def load_degree_maps():
    """Load map from address to node in- and out-degrees..
    """
    with open('data/degree_map.pickle', 'rb') as handle:
        return pickle.load(handle)


def save_sparse_csr(filename, ndarray):
    """Save a sparse array.

    Credit: http://stackoverflow.com/a/8980156.
    """
    np.savez(filename,
             data=ndarray.data,
             indices=ndarray.indices,
             indptr=ndarray.indptr,
             shape=ndarray.shape)


def load_sparse_csr(filename):
    """Load a sparse array.

    Credit: http://stackoverflow.com/a/8980156
    """
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
