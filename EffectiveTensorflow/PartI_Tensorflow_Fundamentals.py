import tensorflow as tf
import numpy as np
# @1.
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
#
# w = tf.get_variable('w', shape=[3, 1])
# f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
# yhat = tf.squeeze(tf.matmul(f, w), 1)
#
# loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)
#
# train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
#
# def generate_data():
#     x_val = np.random.uniform(-10.0, 10.0, size=100)
#     y_val = 5 * np.square(x_val) + 3
#     return x_val, y_val
#
# sess = tf.Session()
#
# sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
#
# for _ in range(1000):
#     x_val, y_val = generate_data()
#     _, loss_val = sess.run([train_op, loss], feed_dict={x:x_val, y:y_val})
#     print(loss_val)
#
# print(sess.run([w]))
#
#


# #### @2
# # # It can be convenient to have a function that returns the static shape when available and dynamic shape when it's not. The following utility function does just that:
# def get_shape(tensor):
#   static_shape = tensor.shape.as_list()
#   dynamic_shape = tf.unstack(tf.shape(tensor))
#   dims = [s[1] if s[0] is None else s[0]
#           for s in zip(static_shape, dynamic_shape)]
#   return dims
#
# # # Now imagine we want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions into one. We can use our get_shape() function to do that:
# #
# # # e.g.
# # b = tf.placeholder(tf.float32, [None, 10, 32])
# # shape = get_shape(b)
# # b = tf.reshape(b, [shape[0], shape[1]*shape[2]])
#
#
# # In fact we can write a general purpose reshape function to collapse any list of dimensions:
# ### @3
#
# import tensorflow as tf
# import numpy as np
#
# def reshape(tensor, dims_list):
#   shape = get_shape(tensor)
#   dims_prod = []
#   for dims in dims_list:
#     if isinstance(dims, int):
#       dims_prod.append(shape[dims])
#     elif all([isinstance(shape[d], int) for d in dims]):
#       dims_prod.append(np.prod([shape[d] for d in dims]))
#     else:
#       dims_prod.append(tf.prod([shape[d] for d in dims]))
#   tensor = tf.reshape(tensor, dims_prod)
#   return tensor
#
#
# b = tf.placeholder(tf.float32, [None, 10, 32])
# print(b)
# b = reshape(b, [0, [1, 2]])
# print(b)


### @4 Scopes and when to use them


# tf.Variable.. assign a_1 as new name...
# a = tf.Variable(1, name='a') # <tf.Variable 'a:0' shape=() dtype=int32_ref>
# b = tf.Variable(1, name='a') # <tf.Variable 'a_1:0' shape=() dtype=int32_ref>

# c = tf.get_variable('b', [1, 1], tf.float32)

# d = tf.get_variable('b', [1, 1], tf.float32) # bug.. Variable b already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

#
# with tf.variable_scope("scope"):
#   a1 = tf.get_variable(name="a", shape=[])
#   a2 = tf.get_variable(name="a", shape=[])  # Disallowed
#
# with tf.variable_scope("scope"):
#   a1 = tf.get_variable(name="a", shape=[])
# with tf.variable_scope("scope", reuse=True):
#   a2 = tf.get_variable(name="a", shape=[])  # OK

# This becomes handy for example when using built-in neural network layers:
# with tf.variable_scope('my_scope'):
#   features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
# # Use the same convolution weights to process the second image:
# with tf.variable_scope('my_scope', reuse=True):
#   features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
#

# Alternatively you can set reuse to tf.AUTO_REUSE which tells TensorFlow to create a new variable if a variable with the same name doesn't exist, and reuse otherwise:
#
# with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
#     features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
#
# with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
#     features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
#


## @5 Broadcasting the good and the ugly
#
# a = tf.constant([[1., 2.], [3., 4.]]) # shape=(2, 2)
#
# b = tf.constant([[1.], [2.]]) # shape=(2,1 )
# bb = tf.tile(b, [1, 2])
# sess = tf.Session()
# print(sess.run(b))
# print(sess.run(bb))
#
# c = a + b # it is identical to c = a + tf.tile(b, [1, 2])
# # print(sess.run(a))
# print(sess.run(b))
# print(sess.run(c))
#


# ## ugly implementation
# a = tf.random_uniform([5, 3, 5])
# b = tf.random_uniform([5, 1, 6])
#
# tiled_b = tf.tile(b, [1, 3, 1])
# c = tf.concat([a, tiled_b], 2)
# d = tf.layers.dense(c, 3, activation=tf.nn.relu)
# ## beautiful implementation
# # But this can be done more efficiently with broadcasting. We use the fact that f(m(x + y)) is equal to f(mx + my). So we can do the linear operations separately and use broadcasting to do implicit concatenation:
# pa = tf.layers.dense(a, 10, activation=None)
# pb = tf.layers.dense(b, 10, activation=None)
# d = tf.nn.relu(pa + pb)
#
#
# # In fact this piece of code is pretty general and can be applied to tensors of arbitrary shape as long as broadcasting between tensors is possible:
#
# def merge(a, b, units, activation=tf.nn.relu):
#     pa = tf.layers.dense(a, units, activation=None)
#     pb = tf.layers.dense(b, units, activation=None)
#     c = pa + pb
#     if activation is not None:
#         c = activation(c)
#     return c
#
# a = tf.constant([[1.], [2.]])
# print(a)
# b = tf.constant([1., 2.])
# print(b)
#
# c = a + b
#
# sess = tf.Session()
# print(sess.run(c)) # [[2. 3.]
#                     #  [3. 4.]]
# #
## 1 2      1  1 #his is because when rank of two tensors don't match, TensorFlow automatically expands the first dimension of the   tensor with lower rank before the elementwise operation, so the result of addition would be [[2, 3], [3, 4]], and the reducing over all parameters would give us 12
# 1 2   +  2  2

# The way to avoid this problem is to be as explicit as possible. Had we specified which dimension we would want to reduce across, catching this bug would have been much easier:


#
# a = tf.constant([[1.], [2.]])
# b = tf.constant([1., 2.])
# c = tf.reduce_sum(a + b, 0)
# sess = tf.Session()
# print(sess.run(c))



## @6 Feeding data to tensorflow

# 1 constants - embed it to tensorflow graph..

# 2. use placeholder to feed:  pass

# 3. use python ops to feed
# def py_input_fn():
#     actual_data = np.random.normal(size=[100])
#     return actual_data
#
# data = tf.py_func(py_input_fn, [], (tf.float32))

# 4. Dataset API --- the recommended way of doing data input
# actual_data = np.random.normal(size=[100])
# dataset = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
# data = dataset.make_one_shot_iterator().get_next()
# print(data)

#  see official site for tfrecord tutorial....


## @7 Take advantage of the overloaded operators

# note: slicing operations like z = x[begin:end] is very inefficient in tensorflow and often better avoid
# e.g.
#
#
# import tensorflow as tf
# import time
#
# x = tf.random_uniform([500, 10])
#
# z = tf.zeros([10])
# # for i in range(500): # this is slow # 0.25 s
# #     z += x[i]
#
# # for x_i in tf.unstack(x): # equivalent, but much faster # 0.026s, 10 times faster
# #     z += x_i
#
# z = tf.reduce_sum(x, axis=0) # this is the most correct way of doing it.. even faster, 0.001919s # another 10 times faster
#
# sess = tf.Session()
# start = time.time()
# sess.run(z)
# print("Took %f seconds." % (time.time() - start))
#
#


#
# z = -x  # z = tf.negative(x)
# z = x + y  # z = tf.add(x, y)
# z = x - y  # z = tf.subtract(x, y)
# z = x * y  # z = tf.mul(x, y)
# z = x / y  # z = tf.div(x, y)
# z = x // y  # z = tf.floordiv(x, y)
# z = x % y  # z = tf.mod(x, y)
# z = x ** y  # z = tf.pow(x, y)
# z = x @ y  # z = tf.matmul(x, y)
# z = x > y  # z = tf.greater(x, y)
# z = x >= y  # z = tf.greater_equal(x, y)
# z = x < y  # z = tf.less(x, y)
# z = x <= y  # z = tf.less_equal(x, y)
# z = abs(x)  # z = tf.abs(x)
# z = x & y  # z = tf.logical_and(x, y)
# z = x | y  # z = tf.logical_or(x, y)
# z = x ^ y  # z = tf.logical_xor(x, y)
# z = ~x  # z = tf.logical_not(x)

# Other operators that aren't supported are equal (==) and not equal (!=) operators which are overloaded in NumPy but not in TensorFlow. Use the function versions instead which are tf.equal and tf.not_equal.


## @8 Understanding order of execution and control dependencies



preferred_order = ['0']