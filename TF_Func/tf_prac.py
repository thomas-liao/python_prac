import tensorflow as tf
import numpy as np


pred = np.arange(36*3).reshape(36, 3)
gt = np.zeros(18*3).reshape(18, 3)
gt = np.vstack([gt, 2*np.ones(18*3).reshape(18, 3)])

pred = tf.convert_to_tensor(pred, dtype=tf.float32)
gt = tf.convert_to_tensor(gt, dtype=tf.float32)


p_1 = tf.gather(pred,[i for i in range(18)])
p_2 = tf.gather(pred, [i for i in range(18, 36)])


gt_1 = tf.gather(gt, [i for i in range(18)])
gt_2 = tf.gather(gt, [i for i in range(18, 36)])



d_pred = tf.sqrt(tf.reduce_sum(tf.square(p_1 - p_2), axis=1))
d_gt = tf.sqrt(tf.reduce_sum(tf.square(gt_1 - gt_2), axis=1))

temp = d_pred - d_gt
temp = tf.square(temp)
loss = tf.reduce_sum(temp)

with tf.Session() as sess:
    print(d_gt.eval())
    print(d_pred.eval())
    print(loss.eval())
    # print(temp.eval())


#
#
# a = np.arange(3)
# b = 2 * np.arange(3)
#
# c = 3 * np.arange(3)
# d = 4 * np.arange(3)
#
# l1 = np.array([a, b])
# l2 = np.array([c, d])
# l1 = tf.convert_to_tensor(l1, dtype=tf.float32)
# l2 = tf.convert_to_tensor(l2, dtype=tf.float32)

# c = tf.gather(l1,[0,1])


# c = tf.sqrt(tf.reduce_sum(tf.squared_difference(a,b)))

# with tf.Session() as sess:
    # print(pred.eval())

