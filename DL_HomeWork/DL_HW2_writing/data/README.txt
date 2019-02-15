##### Problem 2 and Problem 3 #####
Apply your code on the 6 toy datasets, each with 10 positive and 10 negative examples.
We have one pair of 2D points, one pair of 3D points, and one pair 4D points.

Descriptions of data directories:

./data/Data1/data.npz: 2D points
./data/Data2/data.npz: 2D points
./data/Data3/data.npz: 3D points
./data/Data4/data.npz: 3D points
./data/Data5/data.npz: 4D points
./data/Data6/data.npz: 4D points 

#NOTE# All the labels (y) are initialized as 0 for the negative examples and 1 for the positive examples.
For Problem 2, please change the negative labels from 0 to -1 accordingly. For Problem 3, no need to adjust the labels of negative examples.

# The following Python code can be used to load the data for each data set specified by the "dataind".
import numpy as np

dataind = 6
datapath = './data/Data' + str(dataind) + '/data.npz'
data = np.load(datapath)
# x, array of shape [n_samples, n_features]
# y, array of shape [n_samples]
x = data['x']
y = data['y']

# TO-DO #
# You need to add one more dimension of feature to x, where the added dimension are all 1's to introduce a bias term in the parameter theta.
# Write your update strategy for the Problem 2 (Perceptron), Problem 3 (Logistic Regression).

##### Problem 4 #####
Apply your code on a set of data (x, y) pairs.

Descriptions of data:

./Problem4/data.npz: a set of (x, y) data pairs

# The following Python code can be used to load the data
import numpy as np
datapath = './Problem4/data.npz'
data = np.load(datapath)
# x, array of shape [n_samples]
# y, array of shape [n_samples]
x = data['x']
y = data['y']

# TO-DO #
# Try to derive the parameter for y = theta1*x + theta2 given the data points.
# Write your update strategy for Problem 4 (Linear Regression by Mean Square Error using Gradient Descent).
