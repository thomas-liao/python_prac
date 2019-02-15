# # ##### Problem 2 and Problem 3 #####
# # Apply your code on the 6 toy datasets, each with 10 positive and 10 negative examples.
# # We have one pair of 2D points, one pair of 3D points, and one pair 4D points.
# #
# # Descriptions of data directories:
# #
# # ./data/Data1/data.npz: 2D points
# # ./data/Data2/data.npz: 2D points
# # ./data/Data3/data.npz: 3D points
# # ./data/Data4/data.npz: 3D points
# # ./data/Data5/data.npz: 4D points
# # ./data/Data6/data.npz: 4D points
# #
# # #NOTE# All the labels (y) are initialized as 0 for the negative examples and 1 for the positive examples.
# # For Problem 2, please change the negative labels from 0 to -1 accordingly. For Problem 3, no need to adjust the labels of negative examples.
# #
# # # The following Python code can be used to load the data for each data set specified by the "dataind".
# # import numpy as np
# #
# # dataind = 6
# # datapath = './data/Data' + str(dataind) + '/data.npz'
# # data = np.load(datapath)
# # # x, array of shape [n_samples, n_features]
# # # y, array of shape [n_samples]
# # x = data['x']
# # y = data['y']
# #
# # # TO-DO #
# # # You need to add one more dimension of feature to x, where the added dimension are all 1's to introduce a bias term in the parameter theta.
# # # Write your update strategy for the Problem 2 (Perceptron), Problem 3 (Logistic Regression).
# #
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

class LogisticRegression:

    def __init__(self, learning_rate):
        self.theta = None
        self.learning_rate = learning_rate
        self.losses = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # def log_likelihood(self,h, y):
    #     return (-y * np.log(h) - (1-y)*np.log(1-h)).mean()

    def _create_D_Matrix(self, X):
        # most stupid way...
        m = X.shape[0]
        temp = np.zeros((m, m))
        for i in range(X.shape[0]):
            zi = self._sigmoid(np.dot(self.theta, X[i, :]))
            temp[i][i] = zi * (1-zi)
        return temp

    def compute_steps(self, X, y, steps, epsilon_conv = 0.01, diag=False):
        if self.theta is None:
            self.theta = np.zeros(len(x[0]))

        prev_loss = -100

        for step in range(steps):
            # update using 2nd order method: theta^k+1 = theta^k - epsilon * H^-1 g
            Z = np.dot(X, self.theta)
            h = self._sigmoid(Z)

            g = np.dot(X.T, (h-y)) / y.shape[0]

            H = X.T @ self._create_D_Matrix(X) @ X


            if diag:
                eye = np.eye(X.shape[1])
                H_diag = H * eye
                H_r = np.linalg.pinv(H_diag)
            else:
                H_r = np.linalg.pinv(H)   # all Hessian rev

            # print("checkpoint H")
            # print(H)

            # update using 2nd order method: theta^k+1 = theta^k - epsilon * H^-1 g
            epsilon = 0.5
            self.theta -= epsilon * (H_r @ g)

            cur_loss = self.log_likelihood(X, y)
            self.losses.append(cur_loss)

            if np.abs(prev_loss - cur_loss) < epsilon_conv:
                flag = self.compute_miss_all(x, y, step)
                if flag:
                    print("loss converged at step{}, accuracy 100%.".format(step))
                    return
                else:
                    print("loss converged at step{}, but accuracy NOT 100%".format(step))


            flag = self.compute_miss_all(X, y, step)
            # if flag:
            #     print("converged step:", step)
            #     break;

            prev_loss = cur_loss
        print("loss not converged.")

            # if step % 1 == 0:
            #     flag = self.compute_miss_all(x, y, step)
            #     if flag:
            #         break


    def log_likelihood(self, X, y):
        scores = np.dot(x, self.theta)
        pred = self._sigmoid(scores)

        mean_ll = (-y * np.log(pred) - (1-y) * np.log(1-pred)).mean()
        return mean_ll



    def compute_miss_all(self, x, y, i):
        scores = np.dot(x, self.theta)
        # scores = np.dot(x, np.zeros_like(self.theta))
        pred = np.round(self._sigmoid(scores))
        print("Accuracy: {0}".format((pred==y).sum().astype(float) / len(y)))

        if np.all(pred==y):
            print("Converged at step: ", i+1)
            print("converged theta: ", self.theta)
            return True
        return False


dataind = 6

datapath = './data/Data' + str(dataind) + '/data.npz'

def load_logi(datapath=datapath):
    data = np.load(datapath)
    x = data['x']
    y = data['y']
    y = np.array([item if item == 1 else 0 for item in y])
    x = np.hstack([np.ones((len(x), 1)), x])
    return x, y

x, y = load_logi()
learning_rate = 0.01

print(y)
steps =5000
#
logReg = LogisticRegression(learning_rate)

logReg.compute_steps(x, y, steps, diag=False)


plt.plot(logReg.losses)
plt.xlabel("steps")
plt.ylabel("Loss(negative log likelihood)")
plt.title("Loss vs steps for dataset {}".format(dataind))
plt.show()

print(logReg.theta)
#
# #
# def ssigmoid(scores):
#     return 1 / (1 + np.exp(-scores))
# #
# # theta = [-2.31125409e-04 , 5.69917556e-05, -8.37189962e-06,  2.10194507e-04] # 1
# # theta = [-0.02306136,  0.00568655, -0.00083534 , 0.0209729 ]
# theta = [-28.01378592,   2.81208553,  -0.54752278,  25.74551868]
# pred = x @ theta
# pred = ssigmoid(pred)
# print(pred)
# pred = np.round(pred)
# print(pred)
# print(y)
#
# print("sanity check")
# print(np.all(pred == y))
# #


# dataset 1:
# loss converged at step17, accuracy 100%.
# [-2.46867332  0.28445457  2.12688432]
#
# datset 2:
# not converged in 5000 steps.
# [-2.2231039   1.68269178  0.35298951]
#
# # datset 3:
# not converged in 5000 steps.
# [ 1.55321305  0.8646749  -1.18632391 -0.97470241]
#
# dataset 4
# loss converged at step17, accuracy 100%.
# [-1.80083455  0.44237484 -0.06524331  1.63565678]
#
# dataset 5
# loss converged at step18, accuracy 100%.
# [ 0.17144227  1.13502363 -0.31000171 -0.03906323 -0.7169939 ]
#
# dataset 6
# converged in 5000 steps
# [ 0.83886602  0.47546665  0.73281364 -1.61153346 -1.99647899]
