# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:02:09 2022

@author: gitan
"""
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
#creating training data
min_val = -0.6
max_val = 0.6
num_points = 10
#setting seed to 1
np.random.seed(1)
# generating two sets of 10 numbers drawn from the uniform distribution
# setting the numbers to fall between -0.6 and +0.6
set1 = np.random.uniform(min_val, max_val, num_points)
set2 = np.random.uniform(min_val,max_val, num_points)
#reshaping the sets to (10,1)
set1 = set1.reshape(num_points, 1)
set2 = set2.reshape(num_points, 1)
#saving sets in a ndarray(10,2)
input_gitansh = np.append(set1,set2, axis = 1)

# creating output set of size(10,1)
output_gitansh = np.add(set1,set2)

# finding the minimum and maximum values of the dimensions and storing in a variable
dim1 = [min_val, max_val]
dim2 = [min_val, max_val]
#creating a feed forward network of single layer 
nn = nl.net.newff([dim1, dim2], [6,1])
# training the network
error_progress = nn.train(input_gitansh, output_gitansh, show=15, goal=0.00001)
#plotting error progresss graph
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

#pring the results
print('\nTest results:')
data_test = [[0.1,0.2]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])