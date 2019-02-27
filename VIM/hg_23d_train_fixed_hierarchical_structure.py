# Python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import os.path as osp

import tensorflow as tf
import numpy as np
import random
import pdb
import math

import single_key_net_fixed as sk_net
import time
import datetime
from utils import preprocess
# from utils import preprocess_norm

from utils import ndata_tfrecords
from utils import optimizer
from utils import vis_keypoints
from tran3D import quaternion_matrix
from tran3D import polar_to_axis
from tran3D import quaternion_about_axis

from scipy.linalg import logm
import cv2

import hg_utils as ut
import net2d_hg_modified_v1 as hg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('train_log_hg23d_fixed.log')
logger.addHandler(file_handler)


class Hier_loss:
    """Hierarchical structure-aware loss
    lvl1: whole car, 36 key points
    lvl2: (1)front: 9 key points, (2)rear: 8 key poitns, (3)right: left: 18 key points, (4) 18 key points
          (5)top: 6 key points, (6)bottom: 24 key points.
    lvl3: (1)frw(front right wheel), 5 key points (2) rrw(rear right wheel), 5 key poitns, (3) flw, 5 key points
          (4)rlw: 5 key points (5)frp(front right panel): 3 (6)flp: 3 (7)rrp(rear right panel), 5 key points
          (8)rlp: 5 key poitns (9) rrr(rear right roof), 2 key points (10) rlr, 2 key points
          (11)frr, 2 key points (12)flr, 2 keypoints
    *lvl4: individual key points, we have 36 key points.

    """
    def __init__(self):
        self.hierarchical_structure = {}
        # hierarchical level-1: whole car
        self.hierarchical_structure['lvl_1'] = {'whole': [i for i in range(36)], 'count':36}

        # hierarchical level -2: left, right, front, end, top, bottom - 6 parts
        front_indices = [0, 1, 2, 3, 12, 13, 14, 15, 16]
        front_indices = front_indices + [i + 18 for i in front_indices]
        rear_indices = [i for i in range(4, 12)]
        rear_indices.append(17)
        rear_indices = rear_indices + [i + 18 for i in rear_indices]
        right_indices = [i for i in range(18)]
        left_indices = [i + 18 for i in right_indices]

        top_indices = [i for i in range(1, 7)]
        top_indices.extend([i + 18 for i in top_indices])
        bottom_indices = [0] + [i for i in range(7, 18)] + [18] + [i + 18 for i in range(7, 18)]
        print("check")
        print(len(bottom_indices))

        self.hierarchical_structure['lvl_2'] = {'front': front_indices, 'rear': rear_indices, 'left': left_indices,
                                                'right': right_indices, 'top': top_indices, 'bottom': bottom_indices}

        # hierarchical level - 3: small parts: front/rear - left/right wheels (4 parts), left/right front-rear panels (4 parts),
        # left/right - front-rear roofs ( 4 parts), total 12 parts

        # wheels
        front_right_wheel_indices = [i for i in range(12, 17)]
        front_left_wheel_indices = [i + 18 for i in front_right_wheel_indices]
        rear_right_wheel_indices = [8, 9, 10, 11, 17]
        rear_left_wheel_indices = [i + 18 for i in rear_right_wheel_indices]

        # panels
        front_right_panel_indices = [0, 1, 2]
        front_left_panel_indices = [i + 18 for i in front_right_panel_indices]
        rear_right_panel_indices = [5, 6, 7]
        rear_left_panel_indices = [i + 18 for i in rear_right_panel_indices]

        # roofs
        front_right_roof_indices = [2, 3]
        front_left_roof_indices = [20, 21]
        rear_right_roof_indices = [4, 5]
        rear_left_roof_indices = [22, 23]
        self.hierarchical_structure['lvl_3'] = {'frw': front_right_wheel_indices, 'flw': front_left_wheel_indices,
                                                'rrw': rear_right_wheel_indices, 'rlw': rear_left_wheel_indices,
                                                'frp': front_right_panel_indices, 'flp': front_left_panel_indices,
                                                'rrp': rear_right_panel_indices, 'rlp': rear_left_panel_indices,
                                                'frr': front_right_roof_indices, 'flr': front_left_roof_indices,
                                                'rrr': rear_right_roof_indices, 'rlr': rear_left_roof_indices}

    # def hier_structure_loss(self, p_hm, gt_hm, weights):
    #     """Hierarchical loss - tensorflow implementation
    #     Args:
    #         p_hm, gt_hm: predict, groundtruthheat map, each with shape of [batch_size, num_stacks(hg), heat_map_height, heat_map_width, num_key_points]
    #         weights: shape (4,), weights for level-1 (whole_car), level-2(6 parts), level-3 (12 parts), level-4 (36 individual key points)
    #     Returns:
    #         (tf.Tensor) hierarchical loss w.r.t. 4 structure level weighted by given weights
    #     """
    #     # level-1 loss:
    #     loss_lvl_1 = weights[0] * self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_1']['whole'])
    #
    #     # level-2 loss:
    #     loss_lvl_2 = 0.0
    #     keys_lvl_2 = self.hierarchical_structure['lvl_2'].keys()
    #     for k in keys_lvl_2:
    #         loss_lvl_2 += self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_2'][k])
    #     loss_lvl_2 = weights[1] * (1.0 / 6) * loss_lvl_2
    #
    #     # level-3 loss:
    #     loss_lvl_3 = 0.0
    #     keys_lvl_3 = self.hierarchical_structure['lvl_3'].keys()
    #     for k in keys_lvl_3:
    #         loss_lvl_3 += self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_3'][k])
    #     loss_lvl_3 = weights[2] * (1.0 / 12) * loss_lvl_3
    #
    #     # level-4 loss:
    #     loss_lvl_4 = weights[3] * ut._bce_loss(logits=p_hm, gtMaps=gt_hm, name='lvl4_ce_loss', weighted=False)
    #     # loss_lvl_4 = weights[3] * tf.losses.mean_squared_error(predictions=p_hm, labels=gt_hm)
    #
    #     total_loss = loss_lvl_1 + loss_lvl_2 + loss_lvl_3 + loss_lvl_4
    #     return total_loss
    #
    # def group_loss_helper(self, p_hm, gt_hm, group_idx):  ## tensorflow sucks.......
    #     p_hm_group = tf.gather(p_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
    #     p_hm_group = tf.reduce_sum(p_hm_group, axis=-1)
    #     gt_hm_group = tf.gather(gt_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
    #     gt_hm_group = tf.reduce_sum(gt_hm_group, axis=-1)
    #     # return ut._bce_loss(logits=p_hm_group, gtMaps=gt_hm_group, weighted=False) # give negative number right now... TODO: debug
    #     return tf.losses.mean_squared_error(predictions=p_hm_group, labels=gt_hm_group)  # MSE


def read_one_datum(fqueue, dim, key_num=36, hm_dim=64, nStack=4):
  reader = tf.TFRecordReader()
  key, value = reader.read(fqueue)
  basics = tf.parse_single_example(value, features={
    'key2d': tf.FixedLenFeature([key_num * 2], tf.float32),
    'key3d': tf.FixedLenFeature([key_num * 3], tf.float32),
    'image': tf.FixedLenFeature([], tf.string)})

  image = basics['image']
  image = tf.decode_raw(image, tf.uint8)
  image.set_shape([3 * dim * dim])
  image = tf.reshape(image, [dim, dim, 3])

  k2d = basics['key2d']
  k3d = basics['key3d']

  # convert k2d to HMs(4stacks for sHG)
  key_resh = tf.reshape(k2d, shape=(key_num, 2))
  key_resh *= hm_dim  # to match hm_dimension

  key_hm = ut._tf_generate_hm(hm_dim, hm_dim, key_resh)  # tensor, (64, 64, 36), dtype=tf.float32
  # temp = [key_hm for _ in range(nStack)]

  # 2-20-19 build hierarchical structure-aware loss
  temp = []



hier = hier_loss()






  key_hm = tf.stack(temp, axis=0)

  return image, key_hm, k3d

def create_bb_pip(tfr_pool, nepoch, sbatch, mean, shuffle=True):
    if len(tfr_pool) == 3:
        ebs = [int(sbatch * 0.5), int(sbatch * 0.3), sbatch - int(sbatch * 0.5) - int(sbatch * 0.3)]
    elif len(tfr_pool) == 1:
        ebs = [sbatch]
    else:
        print("Input Format is not recognized")
        return

    data_pool = []

    for ix, tfr in enumerate(tfr_pool):
        cur_ebs = ebs[ix]
        tokens = tfr.split('/')[-1].split('_')
        dim = int(tokens[-1].split('.')[0][1:])
        tf_mean = tf.constant(mean, dtype=tf.float32)
        tf_mean = tf.reshape(tf_mean, [1, 1, 1, 3])

        fqueue = tf.train.string_input_producer([tfr], num_epochs=nepoch)
        image, gt_key, gt_3d = read_one_datum(fqueue, dim)

        if shuffle:
            data = tf.train.shuffle_batch([image, gt_key, gt_3d], batch_size=cur_ebs,
                                          num_threads=12, capacity=sbatch * 6, min_after_dequeue=cur_ebs * 3)
        else:
            data = tf.train.batch([image, gt_key, gt_3d], batch_size=cur_ebs,
                                  num_threads=12, capacity=cur_ebs * 5)

        # preprocess input images

        # print("data0]", data[0])
        data[0] = preprocess(data[0], tf_mean) #
        # data[0] = preprocess_norm(data[0]) #

        if ix == 0:
            for j in range(len(data)):
                data_pool.append([data[j]])
        else:
            for j in range(len(data)):
                data_pool[j].append(data[j])

    combined_data = []
    for dd in data_pool:
        combined_data.append(tf.concat(dd, axis=0))
    # print("sanity check : combined_data", combined_data)

    return combined_data


# def vis2d_one_output(image, pred_key, gt_key):
#     image += 128
#     image.astype(np.uint8)
#     dim = image.shape[0]  # height == width by default
#
#     left = np.copy(image)
#     right = np.copy(image)
#
#     pk = np.reshape(pred_key, (36, 2)) * dim
#     gk = np.reshape(gt_key, (36, 2)) * dim
#     pk.astype(np.int32)
#     gk.astype(np.int32)
#
#     for pp in pk:
#         cv2.circle(left, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     for pp in gk:
#         cv2.circle(right, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     final_out = np.hstack([left, right])
#
#     outputs = np.zeros((1, final_out.shape[0], final_out.shape[1], 3), dtype=np.uint8)
#     outputs[0] = final_out
#
#     return outputs

# def vis2d_one_output_hm(image, pred_key, gt_key):
#     # de-norm to 0~255
#     image *= 255
#     # resize to 64*64, to match hm size
#     image = tf.image.resize_images(image, size=(64, 64))
#     image.astype(np.uint8)
#     # print(image)
#     dim = image.shape[0]  # height == width by default,  64 by default
#     # print("dim", dim)
#
#     left = np.copy(image)
#     right = np.copy(image)
#
#     pk = pred_key
#     gk = gt_key
#     pk.astype(np.int32)
#     gk.astype(np.int32)
#
#     for pp in pk:
#         cv2.circle(left, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     for pp in gk:
#         cv2.circle(right, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     final_out = np.hstack([left, right])
#
#     outputs = np.zeros((1, final_out.shape[0], final_out.shape[1], 3), dtype=np.uint8)
#     outputs[0] = final_out
#
#     return outputs

def vis2d_one_output_hm(image, pred_key, gt_key):
    return image


# def add_bb_summary(images, pred_keys, gt_keys, name_prefix, max_out=1):
#     n = images.get_shape().as_list()[0]
#
#     for i in range(np.min([n, max_out])):
#         pred = tf.gather(pred_keys, i)
#         gt = tf.gather(gt_keys, i)
#         image = tf.gather(images, i)
#
#         result = tf.py_func(vis2d_one_output_hm, [image, pred, gt], tf.uint8)
#
#         tf.summary.image(name_prefix + '_result_' + str(i), result, 1)

def add_bb_summary_hm(images, pred_keys_hm, gt_keys_hm, name_prefix, max_out=1):
    n = images.get_shape().as_list()[0]

    for i in range(np.min([n, max_out])):
        # collect from batch
        pred_key_hm = tf.gather(pred_keys_hm, i)
        gt_key_hm = tf.gather(gt_keys_hm, i)
        image = tf.gather(images, i)

        # convert hm to 2d keypoints
        pred_key = ut._hm2kp(pred_key_hm)
        gt_key = ut._hm2kp(gt_key_hm)
        # print("pred_key", pred_key)
        # print("gt_key", gt_key)

        result = tf.py_func(vis2d_one_output_hm, [image, pred_key, gt_key], tf.uint8)

        tf.summary.image(name_prefix + '_result_' + str(i), result, 1)


def eval_one_epoch(sess, val_loss, niters):
    total_loss = .0
    for i in xrange(niters):
        cur_loss = sess.run(val_loss)
        total_loss += cur_loss
    return total_loss / niters


def train(input_tfr_pool, val_tfr_pool, out_dir, log_dir, mean, sbatch, wd):
    """Train Multi-View Network for a number of steps."""
    log_freq = 100
    val_freq = 8000
    model_save_freq = 10000
    tf.logging.set_verbosity(tf.logging.ERROR)

    # maximum epochs
    total_iters = 200001 # smaller test...
    # total_iters = 200000 # batch_size 100
    # total_iters = 1250000 # batchsize = 16
    # lrs = [0.01, 0.001, 0.0001]

    # steps = [int(total_iters * 0.5), int(total_iters * 0.4), int(total_iters * 0.1)]

    # set config file
    config = tf.ConfigProto(log_device_placement=False)
    with tf.Graph().as_default():
        sys.stderr.write("Building Network ... \n")
        # global_step = tf.contrib.framework.get_or_create_global_step() # THIS IS REALLY MESSEED UP WHEN LOADING MODELS..

        # images, gt_key = create_bb_pip(input_tfr_pool, 1000, sbatch, mean, shuffle=True)
        images, gt_keys_hm, gt_3d = create_bb_pip(input_tfr_pool, 1000, sbatch, mean, shuffle=True)

        # print(gt_key.get_shape().as_list()) # key_hm: [B, nStack, h, w, #key_points], i.e. [16, 4, 64, 64, 36]
        # inference model
        #
        # key_dim = gt_key.get_shape().as_list()[1]
        # pred_key = sk_net.infer_key(images, key_dim, tp=True)

        # out_dim = gt_keys_hm.get_shape().as_list()[-1]
        out_dim = 36
        # test_out = sk_net.modified_key23d_64_breaking(images)
        # pred_keys_hm = hg._graph_hourglass(input=images, dropout_rate=0.2, outDim=out_dim, tiny=False, modif=False, is_training=True)

        #preparation with 3d intermediate supervision...
        hg_input, pred_3d = sk_net.modified_hg_preprocessing_with_3d_info_fixed(images, 36 * 3, reuse_=False, tp=False)

        vars_avg = tf.train.ExponentialMovingAverage(0.9)
        vars_to_restore = vars_avg.variables_to_restore()
        # print(vars_to_restore)
        model_saver = tf.train.Saver(vars_to_restore) # when you write the model_saver matters... it will restore up to this point

        train_step = tf.Variable(0, name='train_steps', trainable=False) # you need to move train_step here in order to avoid being loaded

        r3 = tf.image.resize_nearest_neighbor(hg_input, size=[64, 64]) # shape=(16, 64, 64, 256), dtype=float32)

        pred_keys_hm = hg._graph_hourglass_modified_v1(input=r3, dropout_rate=0.2, outDim=out_dim, tiny=False, modif=False, is_training=True) # shape=(16, 4, 64, 64, 36), dtype=float32)

        # Calculate loss
        # total_loss, data_loss = sk_net.L2_loss_key(pred_key, gt_key, weight_decay=wd)
        # train_op, _ = optimizer(total_loss, global_step, lrs, steps)
        k3d_loss = 0.1 * tf.nn.l2_loss(pred_3d - gt_3d)
        k2d_hm_loss = ut._bce_loss(logits=pred_keys_hm, gtMaps=gt_keys_hm, name='ce_loss', weighted=False)
        total_loss = tf.add_n([k3d_loss, k2d_hm_loss])
        init_learning_rate = 2.5e-4 # to be deteremined
        # # exp decay: 125000 / 2000 = 625decays,   0.992658^625 ~=0.01, 0.99^625 ~= 0.00187
        lr_hg = tf.train.exponential_decay(init_learning_rate, global_step=train_step, decay_rate=0.97, decay_steps=2000, staircase=True, name="learning_rate")
        #
        #
        rmsprop_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_hg)

        # disgusting....
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_hg = rmsprop_optimizer.minimize(total_loss, train_step)

        sys.stderr.write("Train Graph Done ... \n")
        #
        # # add_bb_summary_hm(images, pred_keys_hm, gt_keys_hm, 'train', max_out=3) # TODO: enable it
        if val_tfr_pool:
            val_pool = []
            val_iters = []
            accur_pool = []
            for ix, val_tfr in enumerate(val_tfr_pool):
                total_val_num = ndata_tfrecords(val_tfr)
                total_val_iters = int(float(total_val_num) / sbatch) # num of batches, iters / epoch
                val_iters.append(total_val_iters)
                # val_images, val_gt_key = create_bb_pip([val_tfr],
                #                                        1000, sbatch, mean, shuffle=False)
                val_images, val_gt_keys_hm, val_gt_3d = create_bb_pip([val_tfr],
                                                       1000, sbatch, mean, shuffle=False)

                val_r3, val_pred_3d = sk_net.modified_hg_preprocessing_with_3d_info_fixed(val_images, 36 * 3, reuse_=True, tp=False)
                val_r3 = tf.image.resize_nearest_neighbor(val_r3, size=[64, 64])  # shape=(16, 64, 64, 256), dtype=float32)
                # val_pred_key = sk_net.infer_key(val_images, key_dim, tp=False, reuse_=True)

                # val_pred_key = sk_net.infer_key(val_images, key_dim, tp=False, reuse_=True)
                val_pred_keys_hm = hg._graph_hourglass_modified_v1(input=val_r3, outDim=out_dim,is_training=False, tiny=False, modif=False, reuse=True)

                # _, val_data_loss = sk_net.L2_loss_key(val_pred_key, val_gt_key, None)
                val_train_loss_hg = ut._bce_loss(logits=val_pred_keys_hm, gtMaps=val_gt_keys_hm, name="val_ce_loss")
                val_train_loss_3d =  0.1 * tf.nn.l2_loss(val_pred_3d - val_gt_3d)
                val_total_loss = tf.add_n([val_train_loss_3d, val_train_loss_hg])

                # val_pool.append(val_data_loss)
                val_accur = ut._accuracy_computation(output=val_pred_keys_hm, gtMaps=val_gt_keys_hm, nStack=4,
                                                       batchSize=16)

                # val_pool.append(val_train_loss_hg)
                val_pool.append(val_total_loss)
                accur_pool.append(val_accur)
        #
        #         # add_bb_summary(val_images, val_pred_key, val_gt_key, 'val_c' + str(ix), max_out=3)
        #         # add_bb_summary_hm(val_images, val_pred_keys_hm, val_gt_keys_hm, 'val_c' + str(ix), max_out=3) # TODO: argmax pred, draw
            sys.stderr.write("Validation Graph Done ... \n")
        #
        # # merge all summaries
        # # merged = tf.summary.merge_all()
        merged = tf.constant(0)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())



        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sys.stderr.write("Initializing ... \n")
            # initialize graph
            sess.run(init_op)

            model_saver_23d_v1 = tf.train.Saver(max_to_keep=10)


            print('restoring')
            # model_saver.restore(sess, '/home/tliao4/Desktop/new_tf_car_keypoint/tf_car_keypoint/src/log_hg_s4_256/L23d_pmc/model/single_key_4s_hg-85000') # 85k steps
            model_saver.restore(sess, '/home/tliao4/Desktop/tf_car_org/tf_car_keypoint-master/src/L23d_pmc_dropout_bug_fixed/model/single_key-141000')
            print("Successfully restored 3d preprocessing")



            # check
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # for i in 4*range(10):
            #     print("check-img/n", (sess.run(images[1, 60+i, 60+i,:])))
            # print(images)

            model_prefix = os.path.join(out_dir, 'single_key_4s_hg_23d_v1_fix')
            timer = 0
            timer_count = 0
            sys.stderr.write("Start Training --- OUT DIM: %d\n" % (out_dim))
            logger.info("Start Training --- OUT DIM: %d\n" % (out_dim))
            for iter in xrange(total_iters):
                ts = time.time()
                if iter > 0 and iter % log_freq == 0:
                    # print('lr', sess.run(lr_hg))
                    # print('global_step', sess.run(train_step))
                    # key_loss, _, summary = sess.run([data_loss, train_op, merged])
                    key_loss, _, summary = sess.run([total_loss,train_op_hg, merged])

                    # summary_writer.add_summary(summary, i)
                    # summary_writer.flush()

                    sys.stderr.write('Training %d (%fs) --- Key L2 Loss: %f\n'
                                     % (iter, timer / timer_count, key_loss))
                    logger.info(('Training %d (%fs) --- Key L2 Loss: %f\n'
                                     % (iter, timer / timer_count, key_loss)))
                    timer = 0
                    timer_count = 0
                else:
                    # sess.run([train_op])
                    sess.run([train_op_hg])
                    timer += time.time() - ts
                    timer_count += 1

                if val_tfr and iter > 0 and iter % val_freq == 0:
                    cur_lr = lr_hg.eval()
                    print("lr: ", cur_lr)
                    logger.info('lr: {}'.format(cur_lr))

                    sys.stderr.write('Validation %d\n' % iter)
                    logger.info(('Validation %d\n' % iter))
                    # loss
                    for cid, v_dl in enumerate(val_pool):
                        val_key_loss = eval_one_epoch(sess, v_dl, val_iters[cid])
                        sys.stderr.write('Class %d --- Key HM CE Loss: %f\n' % (cid, val_key_loss))
                        logger.info('Class %d --- Key HM CE Loss: %f\n' % (cid, val_key_loss))
                    #
                    for cid, accur in enumerate(accur_pool):
                        rec=[]
                        for i in range(val_iters[cid]):
                             acc = sess.run(accur) # acc: [(float)*36]
                             rec.append(acc)
                        rec = np.array(rec)
                        rec = np.mean(rec, axis=0)
                        avg_accur = np.mean(rec)
                        temp_dict = {}
                        for k in range(36):
                            temp_dict['kp_'+str(iter)] = rec[k]
                        sys.stderr.write('Class %d -- Avg Accuracy : %f\n' %(cid, avg_accur))
                        sys.stderr.write('Classs {} -- All Accuracy:\n{}\n'.format(cid, rec))
                        logger.info('Class %d -- Avg Accuracy : %f\n' %(cid, avg_accur))
                        logger.info('Class {} -- All Accuracy:\n {}\n'.format(cid, rec))




                if iter > 0 and iter % model_save_freq == 0:
                    model_saver_23d_v1.save(sess, model_prefix, global_step=iter)

            model_saver_23d_v1.save(sess, model_prefix, global_step=iter)

            summary_writer.close()
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)


def main(FLAGS):
    assert tf.gfile.Exists(FLAGS.input)
    mean = [int(m) for m in FLAGS.mean.split(',')]
    if tf.gfile.Exists(FLAGS.out_dir) is False:
        tf.gfile.MakeDirs(FLAGS.out_dir)

    with open(osp.join(FLAGS.out_dir, 'meta.txt'), 'w') as fp:
        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        fp.write('train_single_key.py --- %s\n' % dt)
        fp.write('input: %s\n' % FLAGS.input)
        fp.write('weight decay: %f\n' % FLAGS.wd)
        fp.write('batch: %d\n' % FLAGS.batch)
        fp.write('mean: %s\n' % FLAGS.mean)

    log_dir = osp.join(FLAGS.out_dir, 'L23d_pmc')
    if tf.gfile.Exists(log_dir) is False:
        tf.gfile.MakeDirs(log_dir)
    else:
        for ff in os.listdir(log_dir):
            os.unlink(osp.join(log_dir, ff))

    model_dir = osp.join(FLAGS.out_dir, 'model')
    if tf.gfile.Exists(model_dir) is False:
        tf.gfile.MakeDirs(model_dir)

    # train_files = ['syn_car_full_train_d64.tfrecord', 'syn_car_crop_train_d64.tfrecord',
    #                'syn_car_multi_train_d64.tfrecord']
    # val_files = ['syn_car_full_val_d64.tfrecord', 'syn_car_crop_val_d64.tfrecord', 'syn_car_multi_val_d64.tfrecord']
    train_files = ['syn_car_full_train_d64.tfrecord', 'syn_car_paddingmorecrop_train_d64.tfrecord',
                   'syn_car_multi_train_d64.tfrecord']
    val_files = ['syn_car_full_val_d64.tfrecord', 'syn_car_paddingmorecrop_val_d64.tfrecord',
                 'syn_car_multi_val_d64.tfrecord']
    train_files = [osp.join(FLAGS.input, tt) for tt in train_files]
    val_files = [osp.join(FLAGS.input, tt) for tt in val_files]

    train(train_files, val_files, model_dir, log_dir, mean, FLAGS.batch, FLAGS.wd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        type=str,
        default='hg_23d_fixed_2-19',
        help='Directory of output training and L23d_pmc files'
    )
    parser.add_argument(
        '--input',
        type=str,
        # default='/media/tliao4/671073B1329C337D/chi/syn_dataset/tfrecord/car/v1',
        default='/home/tliao4/tliao4/def_car/rigid_car_data/v1',
        help='Directory of input directory'
    )
    parser.add_argument(
        '--mean',
        type=str,
        default='128,128,128',
        help='Directory of input directory'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0,
        help='Weight decay of the variables in network.'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='batch size.'
    )
    parser.add_argument('--debug', action='store_true', help='debug mode')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)