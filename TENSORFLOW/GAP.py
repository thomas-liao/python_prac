import tensorflow as tf
import numpy as np

dim = 5

a = np.ones((dim, dim))
b = 55 * np.ones((dim, dim))
c = 3 * np.ones((dim, dim))

temp = [tf.convert_to_tensor(a), tf.convert_to_tensor(b), tf.convert_to_tensor(c)]

temp = tf.stack(temp, axis=0)

gap = tf.reduce_mean(temp, [1, 2])

with tf.Session() as sess:
    print(sess.run(gap))

