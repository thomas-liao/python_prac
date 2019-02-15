import numpy as np
import matplotlib.pyplot as plt


class MultiClassLogisticRegression:
    def __init__(self, lr=0.001, steps=25000):
        self.lr = lr
        self.steps = steps
        self.iter = 0 # for optimizer

    def _softmax(self, score):
        # avoid saturating
        score -= np.max(score)
        sftmx = (np.exp(score).T / np.sum(np.exp(score), axis=1)).T
        return sftmx

    def one_hot(self, Y):
        m = Y.shape[0]
        temp = np.zeros((Y.shape[0], len(np.unique(Y)))) # assume all class appear at least once

        # most stupid way... any more pythonic way?
        for i in range(len(Y)):
            temp[i, Y[i]] = 1
        return temp

    def get_loss_and_grad(self, x, y, lam):
        # num of examples
        m = x.shape[0]  # First we get the number of training examples
        y_mat = self.one_hot(y)  # Next we convert the integer class coding into a one-hot representation
        scores = np.dot(x, self.theta)  # Then we compute raw class scores given our input and current weights
        prob = self._softmax(scores)  # Next we perform a softmax on these scores to get their probabilities
        loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam / 2) * np.sum(
            self.theta * self.theta)  # We then find the loss of the probabilities
        grad = (-1 / m) * np.dot(x.T, (y_mat - prob)) + lam * self.theta  # And compute the gradient for that loss
        return loss, grad

    def get_prob_and_pred(self, X):
        probs = self._softmax(np.dot(X, self.theta))
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def getAccuracy(self, X, Y):
        prob, pred = self.get_prob_and_pred(X)
        accuracy = sum(pred == Y) / (float(len(Y)))
        return accuracy

    def fit(self, X, y):
        self.theta = np.zeros([X.shape[1], len(np.unique(y))]) # self.theata: [num_feat, num_classes]
        lam = 0 # no regularization...
        self.losses = []

        for i in range(self.steps):
            loss, grad = self.get_loss_and_grad(X, y, lam)
            self.losses.append(loss)
            self.theta -= self.lr * grad

    def get_accuracy(self, X, Y):
        prob, pred = self.get_prob_and_pred(X)
        accuracy = sum(pred==Y) / (float(len(Y)))
        return accuracy


svpath = './data.npz'
data = np.load(svpath)

# X, array of shape [n_samples, n_features]
# Y, array of shape [n_samples]
trainX = data['x']
trainY = data['y']
testX = data['testx']
testY = data['testy']

trainX = np.hstack([np.ones((trainX.shape[0], 1)), trainX])
testX = np.hstack([np.ones((testX.shape[0], 1)), testX])



# x = np.hstack([np.ones((len(x), 1)), x])



# visualize training data

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(trainX[:, 0], trainX[:, 1], c = trainY)
ax1.set_title("Training data visualization")

LR = 0.1
STEPS = 10000
mlr = MultiClassLogisticRegression(lr=LR, steps=STEPS)
#

mlr.fit(trainX, trainY)

# sanity check - loss

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(mlr.losses)
ax2.set_title("Train loss vs steps")


# accuracy on train and test data
print("-----------------------------------------------------------")
print("Learning rate: {}".format(LR))
print("-----------------------------------------------------------")
print("Training steps: {}".format(STEPS))
print("-----------------------------------------------------------")
print("Train accuracy: {}".format(mlr.get_accuracy(trainX, trainY)))
print("-----------------------------------------------------------")
print("Test accuracy: {}".format(mlr.get_accuracy(testX, testY)))
print("-----------------------------------------------------------")
print("final theta:")
print(mlr.theta) #
print("-----------------------------------------------------------")

#[[-1.5743219   0.41412835  0.67010339  0.49009017]
# [ 2.56857327  0.77027842 -0.93972547 -2.39912623]]
plt.show()


