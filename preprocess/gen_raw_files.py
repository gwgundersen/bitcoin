"""Generate raw .npy files for fast loading.
"""

import numpy as np


if __name__ == '__main__':

    train_X = np.loadtxt('data/train_X_raw.txt', dtype=np.int)
    np.save('data/train_X_raw.npy', train_X)

    test_X = np.loadtxt('data/test_X_raw.txt', dtype=np.int)
    np.save('data/test_X_raw.npy', test_X)

    print('Raw .npy files saved.')
