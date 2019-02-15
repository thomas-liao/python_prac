import tensorflow as tf
import numpy as np
# print(tf.__version__)

a = np.ones((2, 1, 5, 5, 3))
a = tf.convert_to_tensor(a, dtype=tf.float32)

b = 2 * np.ones((2, 1, 5, 5, 3))
b = tf.convert_to_tensor(b, dtype=tf.float32)



def group_loss_helper(p_hm, gt_hm, group_idx):
    p_hm_group = tf.gather(p_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
    p_hm_group = tf.reduce_sum(p_hm_group, axis=-1)
    gt_hm_group = tf.gather(gt_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
    gt_hm_group = tf.reduce_sum(gt_hm_group, axis=-1)

    return p_hm_group, gt_hm_group

p, g = group_loss_helper(a, b, [0, 1, 2])

print(p)






# p, g = group_loss_helper(a, b, [0])
#
# print(p)
#
#
with tf.Session() as sess:
    print("")
    print(sess.run(g))

#
#






