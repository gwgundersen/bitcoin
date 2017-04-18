"""Clustering and visualization.
"""

import data
import matplotlib.pyplot as plt
from plot.config import hfont
import matplotlib


matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

plt.style.use('seaborn')

X, _ = data.load('degree')
x = X[:, 0]
y = X[:, 2]
plt.scatter(x, y, color='k', s=5)
plt.xlabel('Address', **hfont)
plt.ylabel('Out degree', **hfont)
#plt.show()
plt.savefig('paper/figures/degree_address.png')
