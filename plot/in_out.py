"""Visualizing the relationship between degrees.
"""

import data
import matplotlib.pyplot as plt
from plot.config import hfont


X_train, X_test = data.load('degree')

x = X_train[:, 1]  # Out degree
y = X_train[:, 2]  # In degree
plt.scatter(x, y, color='k', s=3)
plt.xlabel('Out degree', fontsize=12, **hfont)
plt.ylabel('In degree', fontsize=12, **hfont)
#plt.show()
plt.savefig('in_out_degree.png')