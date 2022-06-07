from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_multi_joint(npoint, radius, nsample, njoint, xyz, points, knn=False, use_xyz=True):
    '''
        Input:
            npoint: int32
            radius: float32
            nsample: int32
            njoint: int32
            xyz: (batch_size, njoint, ndataset, 3) TF tensor
            points: (batch_size, njoint, ndataset, channel) TF tensor, if None will just use xyz as points
            knn: bool, if True use kNN instead of radius search
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Output:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
            idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
            grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
                (subtracted by seed point XYZ) in local regions
        '''
    batch_size = xyz.get_shape()[0].value
    # npoint = xyz.get_shape()[2].value

    xyz_array = tf.split(xyz, batch_size, axis=0)
    if points is not None:
        points_array = tf.split(points, batch_size, axis=0)
    for i in range(batch_size):
        temp_xyz = tf.squeeze(xyz_array[i])
        temp_new_xyz = gather_point(temp_xyz, farthest_point_sample(npoint, temp_xyz))
        #temp_new_xyz = gather_point(temp_xyz, face_point_sample(npoint, temp_xyz))
        if knn:
            _, temp_idx = knn_point(nsample, temp_xyz, temp_new_xyz)
        else:
            temp_idx, pts_cnt = query_ball_point(radius, nsample, temp_xyz, temp_new_xyz)
        temp_grouped_xyz = group_point(temp_xyz, temp_idx)
        # temp_grouped_xyz -= tf.tile(tf.expand_dims(temp_new_xyz, 2), [1, 1, nsample, 1])
        if points is not None:
            temp_points = tf.squeeze(points_array[i])
            temp_grouped_points = group_point(temp_points, temp_idx)
            if use_xyz:
                temp_new_points = tf.concat([temp_grouped_xyz, temp_grouped_xyz], axis=1)
            else:
                temp_new_points = temp_grouped_points
        else:
            temp_new_points = temp_grouped_xyz

        temp_new_xyz = tf.expand_dims(temp_new_xyz, axis=0)
        temp_new_points = tf.expand_dims(temp_new_points, axis=0)
        temp_idx = tf.expand_dims(temp_idx, axis=0)
        temp_grouped_xyz = tf.expand_dims(temp_grouped_xyz, axis=0)
        if i == 0:
            new_xyz = temp_new_xyz
            new_points = temp_new_points
            idx = temp_idx
            grouped_xyz = temp_grouped_xyz
        else:
            new_xyz = tf.concat([new_xyz, temp_new_xyz], axis=0)
            new_points = tf.concat([new_points, temp_new_points], axis=0)
            idx = tf.concat([idx, temp_idx], axis=0)
            grouped_xyz = tf.concat([grouped_xyz, temp_grouped_xyz], axis=0)

    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_selected(npoint, nsample, xyz, points, selected_joint, use_xyz=True):
    '''
        Input:
            npoint: int32
            radius: float32
            nsample: int32
            njoint: int32
            xyz: (batch_size, njoint, ndataset, 3) TF tensor
            points: (batch_size, njoint, ndataset, channel) TF tensor, if None will just use xyz as points
            knn: bool, if True use kNN instead of radius search
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Output:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
            idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
            grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
                (subtracted by seed point XYZ) in local regions
        '''
    batch_size = xyz.get_shape()[0].value
    point_num = xyz.get_shape()[2].value
    joint_num = xyz.get_shape()[1].value

    xyz_array = tf.split(xyz, batch_size, axis=0)
    joint_array = tf.split(tf.expand_dims(selected_joint, axis=2), batch_size, axis=0)
    if points is not None:
        points_array = tf.split(points, batch_size, axis=0)
    for i in range(batch_size):
        temp_xyz = tf.squeeze(xyz_array[i])
        temp_joint = tf.squeeze(joint_array[i], axis=0)

        temp_indx0 = farthest_point_sample(npoint - 1, temp_xyz)
        temp_indx0 = tf.concat([temp_indx0, point_num * tf.ones([joint_num, 1], dtype=tf.int32)], axis=1)
        temp_xyz = tf.concat([temp_xyz, temp_joint], axis=1)

        temp_new_xyz = gather_point(temp_xyz, temp_indx0)
        _, temp_idx = knn_point(nsample, temp_xyz, temp_new_xyz)

        temp_grouped_xyz = group_point(temp_xyz, temp_idx)
        temp_grouped_xyz -= tf.tile(tf.expand_dims(temp_new_xyz, 2), [1, 1, nsample, 1])
        if points is not None:
            temp_points = tf.squeeze(points_array[i])
            temp_grouped_points = group_point(temp_points, temp_idx)
            if use_xyz:
                temp_new_points = tf.concat([temp_grouped_xyz, temp_grouped_xyz], axis=1)
            else:
                temp_new_points = temp_grouped_points
        else:
            temp_new_points = temp_grouped_xyz

        temp_new_xyz = tf.expand_dims(temp_new_xyz, axis=0)
        temp_new_points = tf.expand_dims(temp_new_points, axis=0)
        temp_idx = tf.expand_dims(temp_idx, axis=0)
        temp_grouped_xyz = tf.expand_dims(temp_grouped_xyz, axis=0)
        if i == 0:
            new_xyz = temp_new_xyz
            new_points = temp_new_points
            idx = temp_idx
            grouped_xyz = temp_grouped_xyz
        else:
            new_xyz = tf.concat([new_xyz, temp_new_xyz], axis=0)
            new_points = tf.concat([new_points, temp_new_points], axis=0)
            idx = tf.concat([idx, temp_idx], axis=0)
            grouped_xyz = tf.concat([grouped_xyz, temp_grouped_xyz], axis=0)

    return new_xyz, new_points, idx, grouped_xyz

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                       bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope,
                           bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])
            for j, num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                                padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d' % (i, j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 2, 3, 1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1

def face_point_sample(npoint, xyz):
    '''
    :param npoint: int32
    :param xyz: joint_num * smapled_num * 3
    :return:
     joint_num * npoint
    '''
    Z = -xyz[:,:,2]
    outpoints = tf.nn.top_k(Z, npoint)
    return outpoints[1]

def pointnet_sa_module_multi_joints(xyz, points, npoint, radius, nsample, njoint, mlp, mlp2, group_all, bn_decay, scope,
                       bn=False, pooling='avg', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module with multi joints
        Input:
            xyz: (batch_size, njoint, ndataset, 3) TF tensor
            points: (batch_size, njoint, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            njoint: int32 -- how many joints in each points
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, njoint, npoint, 3) TF tensor
            new_points: (batch_size, njoint, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, njoint, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_multi_joint(npoint, radius, nsample, njoint, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        batch_size = xyz.get_shape()[0].value
        new_points_array = tf.split(new_points, batch_size, axis=0)
        for i in range(batch_size):
            temp_points = tf.squeeze(new_points_array[i])
            for j, num_out_channel in enumerate(mlp):
                temp_points = tf_util.conv2d(temp_points, num_out_channel, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=bn, scope='conv_%d' % (j), bn_decay=bn_decay)
            temp_points = tf.expand_dims(temp_points, axis=0)
            if i == 0:
                new_points = temp_points
            else:
                new_points = tf.concat([new_points, temp_points], axis=0)


        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[3], keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[3], keep_dims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[3], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[3], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay)

        new_points = tf.squeeze(new_points, [3])  # (batch_size, njoint, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_new(pcloud, mlp, bn_decay, scope, bn=False, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    num_point = pcloud.get_shape()[1].value
    with tf.variable_scope(scope) as sc:
        new_points = tf.expand_dims(pcloud, axis=-1)
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        net = tf_util.conv2d(new_points, 64, [1, 3],
                       padding='VALID', stride=[1, 1],
                       bn=bn,
                       scope='conv1', bn_decay=bn_decay,
                       data_format=data_format)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=bn,
                             scope='conv2', bn_decay=bn_decay,
                             data_format=data_format)
        for i, num_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_channel, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=bn,
                                 scope='conv%d' % (i+3), bn_decay=bn_decay,
                                 data_format=data_format)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                        padding='VALID', scope='maxpool')
        # net = tf.squeeze(net)
        return net

def pointnet_sa_new_new(pcloud, mlp, bn_decay, scope, bn=False, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    num_point = pcloud.get_shape()[1].value
    with tf.variable_scope(scope) as sc:
        new_points = tf.expand_dims(pcloud, axis=-1)
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        net = tf_util.conv2d(new_points, 64, [1, 3],
                       padding='VALID', stride=[1, 1],
                       bn=bn,
                       scope='conv1', bn_decay=bn_decay,
                       data_format=data_format)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=bn,
                             scope='conv2', bn_decay=bn_decay,
                             data_format=data_format)
        for i, num_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_channel, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=bn,
                                 scope='conv%d' % (i+3), bn_decay=bn_decay,
                                 data_format=data_format)
        # net = tf_util.max_pool2d(net, [num_point, 1],
        #                                 padding='VALID', scope='maxpool')
        net = tf.squeeze(net)
        return net

def pointnet_sa_module_multi_joints_new(xyz, points, joints, npoint, nsample, njoint, mlp, mlp2, group_all, is_training, bn_decay, scope,
                       bn=True, pooling='avg', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module with multi joints
        Input:
            xyz: (batch_size, njoint, ndataset, 3) TF tensor
            points: (batch_size, njoint, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            njoint: int32 -- how many joints in each points
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, njoint, npoint, 3) TF tensor
            new_points: (batch_size, njoint, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, njoint, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_selected(npoint, nsample, xyz, points, joints, use_xyz)

        # Point Feature Embedding
        batch_size = xyz.get_shape()[0].value
        channel_num = new_points.get_shape()[2].value
        new_points = tf.reshape(new_points, shape=[batch_size, njoint * channel_num, nsample, -1])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay)
        new_points = tf.reshape(new_points, shape=[batch_size, njoint, channel_num, nsample, -1])



        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[3], keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[3], keep_dims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay)

        new_points = tf.squeeze(new_points, [3])  # (batch_size, njoint, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def _group_points(point_cloud, query_cloud, group_size, pc_weight_gt):
    _, num_out_points, _ = query_cloud.shape

    # find nearest group_size neighbours in point_cloud
    _, idx = knn_point(group_size, point_cloud, query_cloud)
    grouped_points = group_point(point_cloud, idx)
    grouped_points_weight = group_point(pc_weight_gt, idx)
    return grouped_points, grouped_points_weight

def _group_points_new(point_cloud, query_cloud, group_size):
    _, num_out_points, _ = query_cloud.shape
    # find nearest group_size neighbours in point_cloud
    _, idx = knn_point(group_size, point_cloud, query_cloud)
    grouped_points = group_point(point_cloud, idx)
    return grouped_points

def _group_points_weights(point_cloud, query_cloud, weights, group_size):
    _, num_out_points, _ = query_cloud.shape
    # find nearest group_size neighbours in point_cloud
    _, idx = knn_point(group_size, point_cloud, query_cloud)
    grouped_points = group_point(point_cloud, idx)
    expand_weights = weights[..., tf.newaxis]
    grouped_weights = group_point(expand_weights, idx)
    return grouped_points, grouped_weights

def _get_distances(grouped_points, query_cloud, group_size, sigma):
    # remove centers to get absolute distances
    deltas = grouped_points - tf.tile(
        tf.expand_dims(query_cloud, 2), [1, 1, group_size, 1]
    )
    dist = tf.reduce_sum(deltas ** 2, axis=3, keepdims=True) / sigma
    return dist

def _get_distances_weights(grouped_points, query_cloud, group_size, sigma, group_weights):
    # remove centers to get absolute distances
    deltas = grouped_points - tf.tile(
        tf.expand_dims(query_cloud, 2), [1, 1, group_size, 1]
    )
    dist = tf.reduce_sum(deltas ** 2, axis=3, keepdims=True) / sigma
    dist = tf.multiply(dist, (1-group_weights))
    return dist

def _get_weight_gt(raw, sample, pc_weights):
    _, idx = knn_point(1, raw, sample)
    sample_weights = group_point(pc_weights, idx)
    sample_weights = tf.squeeze(sample_weights, axis=2)
    return sample_weights
