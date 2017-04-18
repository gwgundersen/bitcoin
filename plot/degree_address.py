"""Clustering and visualization.
"""

import data
import matplotlib.pyplot as plt
from plot.config import hfont


X, _ = data.load('degree')
x = X[:, 0]
y = X[:, 2]
plt.scatter(x, y, color='k', s=3)
plt.xlabel('Address', fontsize=12, **hfont)
plt.ylabel('Out degree', fontsize=12, **hfont)
plt.show()
