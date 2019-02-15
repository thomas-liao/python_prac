##### Problem 3 Multi-Class Logistic Regression with Softmax #####
Apply your code on this toy dataset with 500 training examples and 100 testing examples belonging to 4 classes in total.

The data points are in 2D, so you can easily visualize them in a 2D plane to help you understand
the distribution of those data points. We have 4 classes in total, which are denoted by 0, 1, 2, 3 in Y for each class.

# TO-DO #
Same as the written Homework 1, you need to add one more dimension of feature to X, 
where the added dimension are all 1's to introduce a bias term. It's worth noting that
we have 2 sub-datasets, which are for the training and testing, respectively. Basically, you need to train
your multi-class regression model on the training dataset, and test your model on the testing
dataset. You are asked to try different learning rates and training iterations, and then report 
your best testing accuracy you can achieve on the test dataset.


# Use the following reference code to load the training and testing data
import numpy as np

svpath = './data.npz'
data = np.load(svpath)

# X, array of shape [n_samples, n_features]
# Y, array of shape [n_samples]
trainX = data['x']
trainY = data['y']
testX = data['testx']
testY = data['testy']