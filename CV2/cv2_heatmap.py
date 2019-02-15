import cv2
import numpy as np
import tensorflow as tf


colors = [(241,242,224), (196,203,128), (136,150,0), (64,77,0),
				(201,230,200), (132,199,129), (71,160,67), (32,94,27),
				(130,224,255), (7,193,255), (0,160,255), (0,111,255),
				(220,216,207), (174,164,144), (139,125,96), (100,90,69),
				(252,229,179), (247,195,79), (229,155,3), (155,87,1),
				(231,190,225), (200,104,186), (176,39,156), (162,31,123),
				(210,205,255), (115,115,229), (80,83,239), (40,40,198)]


canvas = tf.zeros((64, 64, 3))


def _tf_makeGaussian(h, w, fwhm=3, center=None):
    x = tf.range(0, w, 1, dtype=tf.float32)
    y = tf.range(0, h, 1, dtype=tf.float32)
    y = tf.reshape(y, shape=(h, 1))

    if center is None:
        x0 = w // 2
        y0 = h // 2
    else:
        x0 = center[0]
        y0 = center[1]

    temp1 = tf.constant(-2.7725887, dtype=tf.float32)
    temp2 = tf.divide((tf.square(x - x0) + tf.square(y - y0)), tf.constant(fwhm**2, tf.float32))

    return tf.exp(tf.multiply(temp1, temp2))

test = _tf_makeGaussian(64, 64)

test = tf.stack([test,]*3, axis=-1)

test *= colors[1]

test = tf.image.resize_images(test, (512, 512))


test2 = _tf_makeGaussian(64, 64, center=[18, 18])
test2 = tf.stack([test2,]*3, axis=-1)
test2 *= colors[5]
test2 = tf.image.resize_images(test2, (512, 512))


combined_test = test + test2

with tf.Session() as sess:
    temp = sess.run(combined_test)
    temp = temp.astype(np.uint8)
    aa = 2
    cv2.imshow('temp',temp)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()