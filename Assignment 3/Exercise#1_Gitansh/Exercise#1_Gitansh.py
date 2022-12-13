# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:53:43 2022

@author: gitan
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)
x = np.random.uniform(-5, 5, 100)
y = [12*i-4 for i in x]
print(y)
plt.scatter(x, y, alpha=0.5)
plt.title("Scatter Plot (X vs Y)")
plt.xlabel("X")
plt.ylabel("y")
np.random.seed(17)
noise = np.random.normal(0, 0.1, 100)
print(noise)
noiseY = [a + b for a, b in zip(y, noise)]
plt.scatter(x, noiseY, alpha=0.5)
plt.title("Scatter Plot (X vs noiseY)")
plt.xlabel("X")
plt.ylabel("y_noise")

#sunil
# import numpy as np
# import matplotlib as mp
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# np.random.seed(13)
# x = np.random.uniform(-1, 1, 100)
# print(np.count_nonzero(x >= 0))
# print(np.count_nonzero(x < 0))
# y = (12*x-4)

# #plot
# mp.pyplot.scatter(x,y,alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('x, y linear')

# y = y + np.random.normal(0.0, 0.1, 100)

# mp.pyplot.scatter(x, y, alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('x, y, linear with noise')