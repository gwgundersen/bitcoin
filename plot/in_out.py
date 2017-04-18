"""Visualizing the relationship between degrees.
"""

import data
import matplotlib.pyplot as plt
from plot.config import hfont


plt.style.use('seaborn')

X_train, X_test = data.load('degree')

x = X_train[:, 1]  # Out degree
y = X_train[:, 2]  # In degree
plt.scatter(x, y, color='k', s=20)
plt.xlabel('Outdegree', **hfont)
plt.ylabel('Indegree', **hfont)
#plt.show()
plt.savefig('paper/figures/in_out_degree.png')
