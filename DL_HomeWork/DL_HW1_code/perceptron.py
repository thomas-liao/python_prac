# # # ##### Problem 2 and Problem 3 #####
# # # Apply your code on the 6 toy datasets, each with 10 positive and 10 negative examples.
# # # We have one pair of 2D points, one pair of 3D points, and one pair 4D points.
# # #
# # # Descriptions of data directories:
# # #
# # # ./data/Data1/data.npz: 2D points
# # # ./data/Data2/data.npz: 2D points
# # # ./data/Data3/data.npz: 3D points
# # # ./data/Data4/data.npz: 3D points
# # # ./data/Data5/data.npz: 4D points
# # # ./data/Data6/data.npz: 4D points
# # #
# # # #NOTE# All the labels (y) are initialized as 0 for the negative examples and 1 for the positive examples.
# # # For Problem 2, please change the negative labels from 0 to -1 accordingly. For Problem 3, no need to adjust the labels of negative examples.
# # #
# # # # The following Python code can be used to load the data for each data set specified by the "dataind".
# # # import numpy as np
# # #
# # # dataind = 6
# # # datapath = './data/Data' + str(dataind) + '/data.npz'
# # # data = np.load(datapath)
# # # # x, array of shape [n_samples, n_features]
# # # # y, array of shape [n_samples]
# # # x = data['x']
# # # y = data['y']
# # #
# # # # TO-DO #
# # # # You need to add one more dimension of feature to x, where the added dimension are all 1's to introduce a bias term in the parameter theta.
# # # # Write your update strategy for the Problem 2 (Perceptron), Problem 3 (Logistic Regression).
# # #
# # # ##### Problem 4 #####
# # # Apply your code on a set of data (x, y) pairs.
# # #
# # # Descriptions of data:
# # #
# # # ./Problem4/data.npz: a set of (x, y) data pairs
# # #
# # # # The following Python code can be used to load the data
# # # import numpy as np
# # # datapath = './Problem4/data.npz'
# # # data = np.load(datapath)
# # # # x, array of shape [n_samples]
# # # # y, array of shape [n_samples]
# # # x = data['x']
# # # y = data['y']
# # #
# # # # TO-DO #
# # # # Try to derive the parameter for y = theta1*x + theta2 given the data points.
# # # # Write your update strategy for Problem 4 (Linear Regression by Mean Square Error using Gradient Descent).
# # #
#
import numpy as np
import matplotlib.pyplot as plt


#  # just for fun....
# def compute_one_step_with_whole_batch(x, y, learning_rate = 0.1, theta=theta):
#     f = theta @ (x.T)
#     pred = np.array([1 if item > 0 else -1 for item in f]) # prediction
#
#     correctness = pred * y
#     missclassified = [False if item > 0 else True for item in correctness]
#     print('miss_classified', np.sum(missclassified))
#
#     all_grad = (x.T) * y
#     all_grad = np.array(all_grad)
#
#     grad_used = all_grad[:, missclassified]
#     avg_grad = np.mean(grad_used, axis=1)
#     theta += learning_rate * avg_grad
import numpy as np

class Perceptron:

    def __init__(self, learning_rate):
        self.theta = None
        self.learning_rate = learning_rate

    def compute_one_step(self, x, y, idx):
        if self.theta is None:
            self.theta = np.zeros(len(x[0]))

        print('step: ', idx)
        f = np.dot(self.theta, x[idx % len(x)])
        pred = 1 if f > 0 else -1

        self.theta = self.theta + self.learning_rate * (y[idx % len(x)] - pred) * x[idx % len(x)] # update method 1
        if self.compute_miss_all(x, y):
            return True
        return False

    def compute_miss_all(self, x, y):

        pred = x @ self.theta

        pred = [1 if item > 0 else -1 for item in pred]

        is_converged = np.all(pred == y)
        if is_converged:
            print("converged theta is:", self.theta)
        return is_converged

dataind = 5
datapath = './data/Data' + str(dataind) + '/data.npz'
def load_perc(datapath=datapath):
    data = np.load(datapath)
    x = data['x']
    y = data['y']
    y = np.array([item if item == 1 else -1 for item in y])
    x = np.hstack([np.ones((len(x), 1)), x])
    return x, y


x, y = load_perc()
p = Perceptron(0.1)
steps = 5000
for i in range(steps):
    if p.compute_one_step(x, y, i) :
        print("Converged at step: {}!".format(i))
        break
    if i == steps - 1:
        print('Not converged in {} steps'.format(steps))








# sanity check
# theta = [0.02, 0.3349, -0.08, 0.02, -0.2169] # gt - 5
# theta = [ 0.2       ,  1.36027321, -0.6223659 , -0.32965719 ,-1.42771634]
# theta = [0.4    ,     1.57493834, -0.35980074,  0.10937016, -1.34900952]
# theta = [ 0.2 ,        1.36027321, -0.6223659 , -0.32965719, -1.42771634]

# theta = [ 0.4  ,       1.57493834, -0.35980074,  0.10937016, -1.34900952]
#
# pred = x @ theta
#
#
# pred = [1 if item > 0 else -1 for item in pred]
# print(pred)
# print(y)
#
#
# print("is_converged")
# print(np.all(pred == y))