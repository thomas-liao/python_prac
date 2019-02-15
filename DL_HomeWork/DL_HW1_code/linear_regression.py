# # ##### Problem 4 #####
# # Apply your code on a set of data (x, y) pairs.
# #
# # Descriptions of data:
# #
# # ./Problem4/data.npz: a set of (x, y) data pairs
# #
# # # The following Python code can be used to load the data
# # import numpy as np
# # datapath = './Problem4/data.npz'
# # data = np.load(datapath)
# # # x, array of shape [n_samples]
# # # y, array of shape [n_samples]
# # x = data['x']
# # y = data['y']
# #
# # # TO-DO #
# # # Try to derive the parameter for y = theta1*x + theta2 given the data points.
# # # Write your update strategy for Problem 4 (Linear Regression by Mean Square Error using Gradient Descent).
# #

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, learning_rate):
        self.theta = None
        self.learning_rate = learning_rate

    def compute_steps(self, x, y, steps):
        if self.theta is None:
            # self.theta = np.random.rand(2) if len(x.shape) == 1 else np.random.rand(x.shape[1]) # already pre-pad 1 as bias
            self.theta = np.zeros(2)
        for step in range(steps):
            pred = np.dot(x, self.theta)
            err = pred - y
            avg_cost = 1 / (2*len(y)) * np.dot(err.T, err)
            self.theta = self.theta - (self.learning_rate * (1 / len(y)) * np.dot(x.T, err))
            if step % 50 == 0:
                print("Step: {}, Avg_cost: {} ".format(step, avg_cost))
            if step == steps-1:
                print("Final theta", self.theta)

datapath = './data/Problem4/data.npz'

def load_linear_reg(datapath=datapath):
    data = np.load(datapath)
    x = data['x']
    y = data['y']
    x = np.hstack([np.ones((len(x), 1)), x[:, np.newaxis]])
    return x, y

x, y = load_linear_reg()
print(y)
# print('checky')
# print(y)


model = LinearRegression(0.01)

model.compute_steps(x, y, 1000)







