import numpy as np
import tensorflow as tf


a = tf.convert_to_tensor(np.ones((1, 4, 64, 64, 36)), dtype=tf.float32)

b = tf.convert_to_tensor(1.01*np.ones((1, 4, 64, 64, 36)), dtype=tf.float32)

loss_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=b))

loss_mse = tf.nn.l2_loss(a-b)

loss_mse2 = tf.losses.mean_squared_error(labels=a, predictions=b)

with tf.Session() as sess:
    print(sess.run(loss_mse))
    print(sess.run(loss_mse2) * 1*4*64*64*36/2)

