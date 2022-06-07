import ITOP_512 as setting
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from Regression_3D.utils import tf_util
from transform_nets import input_transform_net, feature_transform_net, input_transform_net_3d, feature_transform_net_3d

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    joint_num = setting.joint_num
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    # print(point_feat)

    net = tf_util.conv2d(point_feat, 64, [1, 1],  #BxNx1x64
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool')
    # print(global_feat)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])  #BxNx1x1024

    # concat_feat = tf.concat(3, [point_feat, global_feat_expand])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)    #BxNx1x1088
    # print(concat_feat)

    net = tf_util.conv2d(concat_feat, 512, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 50, [1, 1],  #BxNx1x50
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2]) # BxNxC

    # add the following part
    net = tf.reshape(net, [batch_size, -1])           #Bx50N
    net = tf_util.fully_connected(net, setting.joint_num * 3, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf.reshape(net, [batch_size, setting.joint_num, 3])

    return net, end_points


def get_every_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    # batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    batch_size = 5
    num_point = 15
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])                #BxNx1x64
    # print(point_feat)

    net = tf_util.conv2d(point_feat, 64, [1, 1],  # BxNx1x64
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],  # BxNx1x128
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],  #BxNx1x1024
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point, 1],  #Bx249x1x1024
                                     padding='VALID', scope='maxpool')
    # print(global_feat)

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1]) #Bx3735x1x1024
    # concat_feat = tf.concat(3, [point_feat, global_feat_expand])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)  #
    # print(concat_feat)

    net = tf_util.conv2d(concat_feat, 512, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 50, [1, 1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2]) # BxNxC

    # add the following part
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, setting.joint_num * 3, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf.reshape(net, [batch_size, setting.joint_num, 3])

    return net, end_points, input_image


def get_loss(pred, pose_3d_init, pose_3d_gt):
    pose = tf.add(pred, pose_3d_init, name='pose_refine')
    weights=np.array([1/45,1/45,2/15,2/15,2/15,2/15,2/15,2/15,1/45,1/45,1/45,1/45,1/45,1/45,1/45])
    weights = np.expand_dims(weights,axis=1)
    weights_tensor = tf.constant(weights,dtype=tf.float32)
    # loss_alpha = tf.reduce_mean(tf.norm(tf.multiply((pose-pose_3d_gt), setting.w_matrix)))
    loss_alpha = tf.reduce_mean(tf.matmul(tf.norm(pose - pose_3d_gt,axis=2),weights_tensor))
    # loss_beta = tf.reduce_mean(tf.norm(tf.matmul(pose, rot_mat)-pose_2d_gt))
    # loss_beta = tf.reduce_mean(tf.norm(tf.square(tf.multiply(pred, rot_mat) - pose_2d_gt)))

    # loss = loss_alpha + reg_weight * loss_beta
    loss = loss_alpha

    return loss, loss_alpha


def get_model_3d(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxJxNx3, output BxJxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    joint_num = point_cloud.get_shape()[1].value
    num_point = point_cloud.get_shape()[2].value

    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net_3d(point_cloud, is_training, bn_decay, K=3)    #    point_cloud.shape=BxJxNx3  transform.shape=BxJx3x3
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)                  #  BxJxNx3x1

    net = tf_util.conv3d(input_image, 64, [1, 1, 3],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 64, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)             #  BxJxNx1x64

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net_3d(net, is_training, bn_decay, K=64)    #BxJx64x64
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[3]), transform)   #  BxJxNx64
    point_feat = tf.expand_dims(net_transformed, [3])                   #  BxJxNx1x64
    # print(point_feat)

    net = tf_util.conv3d(point_feat, 64, [1, 1, 1],  #BxJxNx1x64
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 128, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 1024, [1, 1, 1],  #BxJxNx1x1024
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    
    global_feat = tf_util.max_pool3d(net, [1, num_point, 1],  #BxJx1x1x1024
                                     padding='VALID', scope='maxpool', stride=[1,2,2])
    # print(global_feat)
    global_feat_expand = tf.tile(global_feat, [1, 1, num_point, 1, 1])  #BxJxNx1x1024

    # concat_feat = tf.concat(3, [point_feat, global_feat_expand])
    concat_feat = tf.concat([point_feat, global_feat_expand], 4)    #BxJxNx1x1088
    # print(concat_feat)

    net = tf_util.conv3d(concat_feat, 512, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 256, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 128, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 128, [1, 1, 1],
                         padding='VALID', stride=[1,1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv3d(net, 50, [1, 1, 1],  #BxJxNx1x50
                         padding='VALID', stride=[1,1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [3]) # BxNxC

    # add the following part
    net = tf.reshape(net, [batch_size, -1])           #Bx50NJ
    net = tf_util.fully_connected(net, setting.joint_num * 3, bn=False, is_training=is_training, scope='tfc1', bn_decay=bn_decay, activation_fn=None)
    net = tf.reshape(net, [batch_size, setting.joint_num, 3])

    return net, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,512,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
