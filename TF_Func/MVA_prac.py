# import tensorflow as tf
# import numpy as np
#
# a = tf.Variable(1.0, trainable=True, name='Thomas_0')
# # b = tf.Variable(2, trainable=False, name="Thomas_useless")
# global_step = tf.contrib.framework.get_or_create_global_step()
#
#
# op = tf.assign(a, 2*a, name='Thomas_assign')
# #
# var_avg = tf.train.ExponentialMovingAverage(0.9, global_step)
# var_avg_op = var_avg.apply(tf.trainable_variables())
# temp = tf.trainable_variables()
# with tf.control_dependencies([op, var_avg_op]):
#     useless_op = tf.no_op(name='haha')
#
#
#
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#
#
#
#
#
#
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(temp)
#     for i in range(10):
#         res, _, _ = sess.run([a, op, var_avg_op])
#         print(res)
#         # a = 1
#
#
#
#

#
# # test2
import tensorflow as tf
import numpy as np
#
# a = tf.Variable(1.0, name='a')
# b = tf.Variable(2.0, name='b')
# c = tf.Variable(3.0, name='c', trainable=False)
#
# ema = tf.train.ExponentialMovingAverage(decay=0.9)
# ema.apply([a, b])
#
#
#
# with tf.Session() as sess:
#     print(tf.get_default_graph().get_all_collection_keys())
#     print(tf.get_default_graph().get_collection('moving_average_variables'))
#     print(ema.variables_to_restore())



# test3 - full combination - template for ema for weights

# def learning rate
# def optimizer
# def training operation i.e. with tf.control_dependencies(update_ops): pt = tf.train.minimize...
update_ops = tf.get_collections(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    opt = tf.train.RMSPropOptimizer(learning_rate=2.54)




