from Regression_3D.utils import tf_util
import numpy as np
import tensorflow as tf
import ITOP_512 as setting
from pointnet_util import pointnet_sa_module, pointnet_sa_module_multi_joints

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, bn_decay=None, batch_size=None, points=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    njoint = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    if points is not None:
        l0_points = points
    else:
        l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layer
    l1_xyz, l1_points, l1_indices = pointnet_sa_module_multi_joints(l0_xyz, l0_points, npoint=256, radius=0.05, nsample=32, njoint=njoint, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_multi_joints(l1_xyz, l1_points, npoint=64, radius=0.1, nsample=32, njoint=njoint, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_multi_joints(l2_xyz, l2_points, npoint=16, radius=0.2, nsample=32, njoint=njoint, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, activation_fn=None)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay, activation_fn=None)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, njoint * 3, bn=True, is_training=is_training, scope='tfc1', activation_fn=None)
    net = tf.reshape(net, [-1, setting.joint_num, 3])

    return net

def get_loss_direct(pred, pose_3d_gt):
    batch_size = pose_3d_gt.get_shape()[0].value
    njoint = pose_3d_gt.get_shape()[1].value

    weights = np.array([1 / 45, 1 / 45, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45])
    weights = np.expand_dims(weights, axis=1)
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    channel_weight = np.tile(np.array([[10], [10], [1]], dtype=np.float32),[batch_size, 1, njoint])
    channel_weight_tensor = tf.constant(channel_weight, dtype=tf.float32)

    pose = pred

    # loss = tf.reduce_mean(tf.matmul(tf.norm(tf.matmul(tf.subtract(pose, pose_3d_gt), channel_weight_tensor), axis=2), weights_tensor), name='loss')
    loss = tf.reduce_mean(tf.matmul(tf.norm(tf.subtract(pose, pose_3d_gt), axis=2), weights_tensor), name='loss')
    return loss


def get_loss(pred, pose_3d_init, pose_3d_gt):
    batch_size = pose_3d_gt.get_shape()[0].value
    njoint = pose_3d_gt.get_shape()[1].value

    weights = np.array([1 / 45, 1 / 45, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 2 / 15, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45])
    weights = np.expand_dims(weights, axis=1)
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    channel_weight = np.tile(np.array([[10], [10], [1]], dtype=np.float32),[batch_size, 1, njoint])
    channel_weight_tensor = tf.constant(channel_weight, dtype=tf.float32)

    pose = tf.add(pred, pose_3d_init, name='pose_refine')

    # loss = tf.reduce_mean(tf.matmul(tf.norm(tf.matmul(tf.subtract(pose, pose_3d_gt), channel_weight_tensor), axis=2), weights_tensor), name='loss')
    loss = tf.reduce_mean(tf.matmul(tf.norm(tf.subtract(pose, pose_3d_gt), axis=2), weights_tensor), name='loss')
    # loss = tf.reduce_mean(tf.norm(tf.subtract(pose, pose_3d_gt), axis=2), name='loss_alpha')

    return loss

def get_loss_new(pred, pose_3d_gt, pose_2d_gt, rot_mat, bbox_center_tensor, setting, weight_beta=2):
    batch_size = pose_3d_gt.get_shape()[0].value
    njoint = pose_3d_gt.get_shape()[1].value

    # From normalized coordinate to global coordinate
    bbox_size = np.transpose(np.tile(setting.bbox_size, [batch_size, njoint, 1]), [0, 2, 1])
    bbox_size = np.tile(setting.bbox_size, [njoint, 1])
    bbox_size_tensor = tf.constant(bbox_size, dtype=tf.float32)
    pred_3d = tf.add(tf.multiply(pred, bbox_size_tensor), bbox_center_tensor)

    loss_alpha = tf.reduce_mean(tf.norm(tf.subtract(pred, pose_3d_gt), axis=2), name='loss_alpha')
    loss_beta = tf.reduce_mean(tf.norm(tf.subtract(tf.matmul(pred_3d, rot_mat), pose_2d_gt), axis=1), name='loss_beta') * weight_beta

    loss = loss_alpha + loss_beta

    return loss, loss_alpha, loss_beta

def get_loss_pretrain(pred, pose_3d_gt):
    batch_size = pose_3d_gt.get_shape()[0].value
    njoint = pose_3d_gt.get_shape()[1].value

    # From normalized coordinate to global coordinate
    # bbox_size = np.transpose(np.tile(setting.bbox_size, [batch_size, njoint, 1]), [0, 2, 1])
    # bbox_size = np.tile(setting.bbox_size, [njoint, 1])
    # bbox_size_tensor = tf.constant(bbox_size, dtype=tf.float32)
    # pred_3d = tf.add(tf.multiply(pred, bbox_size_tensor), bbox_center_tensor)

    loss_alpha = tf.reduce_mean(tf.norm(tf.subtract(pred, pose_3d_gt), axis=2, ord=2), name='loss_alpha')
    # loss_beta = tf.reduce_mean(tf.norm(tf.subtract(tf.matmul(pred_3d, rot_mat), pose_2d_gt), axis=1), name='loss_beta') * weight_beta

    loss = loss_alpha

    return loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)