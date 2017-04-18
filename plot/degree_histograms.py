"""Plot histogram of in-degree and out-degree nodes.
"""

import data
from plot.config import hfont
import matplotlib.pyplot as plt
import numpy as np
import sys


N_ADDRESSES = 444075
BIN_SIZE    = 0.1


def log(x):
    x = x.astype(np.float)
    x[x == 0] = 0.0001
    assert (x == np.inf).any() == False
    x = np.log10(x)
    return x


def plot(x, y_label):
    # Note: scale the bins appropriately according to whichever log we use.
    #
    # Max out degree: 164333
    # Max in degree : 172727
    #
    # >>> np.log(172727)
    # 12.059467592389684
    # >>> np.log10(172727)
    # 5.2373602300659332
    # >>> np.log2(172727)
    # 17.398134091301763

    bins = range(0, 173000, 5000)
    plt.hist(x, bins=bins, color='k')
    plt.yscale('log')
    plt.xlabel('Address', **hfont)
    plt.ylabel(y_label, **hfont)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    X_train, _ = data.load(ds_type='degree')
    if sys.argv[1] == 'out':
        label = 'Out degree'
        x = X_train[:, 1]
    else:
        label = 'In degree'
        x = X_train[:, 2]
    plot(x, label)

