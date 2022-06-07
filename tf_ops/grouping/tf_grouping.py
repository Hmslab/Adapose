import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))
def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')
def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    # print(b, n, c, m)
    # print(xyz1, (b,1,n,c))
    # xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz1 = tf.tile(xyz1[:, tf.newaxis, ...], [1, m, 1, 1])
    xyz2 = tf.tile(xyz2[..., tf.newaxis, :], [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    # print(dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    # print(idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

if __name__=='__main__':
    

    xyz1 = tf.placeholder(shape=[1, 5, 3], name='asd', dtype=tf.float32)

    sess = tf.InteractiveSession()
    import numpy as np
    feed_data = np.array([[0.1, 0, 0], [0.14, 0, 0], [0.3, 0, 0], [0.4, 0, 0], [0.5, 0, 0]])
    feed_data = feed_data[np.newaxis, ...]
    gt = tf.constant([3, 3, 3, 3, 3], dtype=tf.int32)

    #
    idx, pts_cnt = query_ball_point(0.15, 3, xyz1, xyz1)  # idx shape [B,np,ns]
    not_core = tf.cast(tf.not_equal(pts_cnt, gt), tf.float32)  # core point is 0, not core point is 1 shape=[B, np]
    not_cor_idx = tf.where(not_core)  # shape = (?, 2)
    batch_idx = not_cor_idx[:, 0:1]
    # no_core_num = tf.shape(batch_idx)[0]
    batch_idx = tf.tile(batch_idx, [1, 3])
    batch_idx = tf.reshape(batch_idx, [-1])
    neighbor = tf.gather_nd(idx, not_cor_idx)  # (?, ns)
    neighbor = tf.cast(tf.reshape(neighbor, [-1]), tf.int64)
    new_idx = tf.concat([batch_idx[:, tf.newaxis], neighbor[:, tf.newaxis]], axis=1)
    data = tf.gather_nd(not_core, new_idx)  # ?*ns
    data = tf.cast(tf.reshape(data, [-1, 3]), tf.float32)
    loss = tf.reduce_mean(tf.reduce_prod(data, axis=1))

    print(sess.run(not_core, feed_dict={xyz1: feed_data}))
    print(sess.run(loss, feed_dict={xyz1: feed_data}))
    print(sess.run(data, feed_dict={xyz1: feed_data}))
    








