import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # kill warnings.. does not seem to be a good practice
import tensorflow as tf
import numpy as np

num_samples = 100
# toy example
x1 = np.arange(num_samples).reshape(num_samples)
x1 = x1.astype(np.float32)
y1 = 2 * x1 + 2 # some noise


x2 = 2 * np.arange(num_samples).reshape(num_samples)
x2 = x2.astype(np.float32)
y2 = 2 * x2 + 3  # add some noise

# learn paramters to map (x1, x2) - > (y1, y2)

X = np.vstack([x1, x2]).transpose()
Y = np.vstack([y1, y2]).transpose()


x_in = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_in = tf.placeholder(dtype=tf.float32, shape=(None, 2))

# building graph
with tf.variable_scope('MLP'):
    pred = tf.layers.dense(x_in, units=4, activation=tf.nn.relu)
    pred = tf.layers.dense(pred, units=2, activation=tf.nn.relu)

with tf.variable_scope('loss'):
    loss = tf.nn.l2_loss(pred - y_in)
step = tf.Variable(0, trainable=False)


init_lr = 0.00025
lr = tf.train.exponential_decay(init_lr, global_step=step, decay_steps=2000, decay_rate=0.95)
optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)

train_op = optimizer.minimize(loss)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


model_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    model_saver.restore(sess, 'example-99000')
    print(sess.run(pred, feed_dict={x_in:[[1,2]]}))


