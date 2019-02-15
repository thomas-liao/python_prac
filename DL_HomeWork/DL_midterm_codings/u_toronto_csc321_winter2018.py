import numpy as np

X = np.arange(15).reshape(5,3)
w = np.zeros((3, 1))

t = np.array([1,2,3,4,5])
t = np.reshape(t, (5, 1))

y_bar = np.dot(X, w) - t

# single: dL/ dw.T = (yi - ti) * xi - > dl / dw = (yi - ti) * xi.T
# multiple - > = y_bar = (np.dot(X, w) - t)
# res = np.dot(X.T, y_bar) / N

w_bar = np.dot(X.T, y_bar) / X.shape[0]

alpha = 0.01

w -= alpha * w_bar
print(w)

b_bar = np.mean(y_bar)
b -= alpha * b_bar
