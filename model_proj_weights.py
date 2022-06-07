'''
My Model
Given the sampled point cloud(N * J * 256 * 3), trained a model that output the 3D joint coordinate(N * J * 3) and the
predicted 2D joint heat map
'''
import sys
import os

from matplotlib.pyplot import axis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/distance'))
import cv2
import time
import copy
import utils.np_utils as utils
from utils import tf_util
import datetime
import importlib
import numpy as np
import scipy.io as io
import tensorflow as tf
from tf_nndistance import nn_distance
from tf_grouping import query_ball_point, knn_point, group_point
from pointnet_util import pointnet_sa_module, pointnet_sa_module_multi_joints, _group_points_weights, _get_distances_weights, pointnet_sa_new
from DatasetReader import DatasetReader as dataset
from tf_util import encoder_with_convs_and_symmetry, decoder_with_fc_only, fully_connected


def tic():
    globals()['tt'] = time.perf_counter()

def toc():
    print('Elapsed time: %.8f seconds' % (time.perf_counter() - globals()['tt']))

class Trainer(object):
    def __init__(self, args, sess, mode='train'):
        '''
        Input:

        '''
        self.mode = mode
        # path
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.model_path = args.save_path
        self.setting = importlib.import_module(args.setting)

        # parameters
        self.sess = sess
        self.window_size = self.setting.window_size
        self.g_lr = args.g_lr
        
        self.is_denoising = False
        

        self.MAXITER = args.max_iteration
        bonein = self.setting.bonearray1
        self.boneindex = tf.expand_dims(tf.constant(bonein,name='boneindex',dtype=tf.float32),axis=0)
        self.boneindex = tf.tile(self.boneindex,[self.setting.batch_size*self.setting.window_size,1,1])
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.load_model = True
        else:
            self.load_model = False
        self.weight_3d = [100, 10, 1e-3]

    

        # data size
        self.batch_size = self.setting.batch_size
        self.njoint = self.setting.joint_num
        self.nsampled = self.setting.sample_pixel_total
        self.simplified_num = self.setting.simplified_num

        # data
        print('{}-Preparing datasets...'.format(datetime.datetime.now()))
        self.train_set = dataset(self.data_path+'ITOP_side_train_',self.setting, is_shuffle=True)
        self.val_set = dataset(self.data_path+'ITOP_side_test_',self.setting, is_shuffle=False)
        

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
       
        self.build_model(mode=self.mode)

        # log
        self.train_summary_writer = tf.summary.FileWriter(self.model_path + 'train', self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(self.model_path + 'valid')
        self.sv = tf.train.Supervisor(logdir=self.model_path,
                                      global_step=self.global_step,
                                      saver=self.saver,
                                      summary_writer=self.train_summary_writer)
        


    def build_model(self, use_weight=True, mode='train'):
        '''
        :return:
        '''
        # define place holder
        self.pl_is_training = tf.placeholder(tf.bool, shape=[])
        self.pl_3d_gt_pose = tf.placeholder(tf.float32, shape=[self.batch_size*self.window_size, self.njoint, 3], name='3d_gt')
        self.pl_2d_gt_pose = tf.placeholder(tf.float32, shape=[self.batch_size*self.window_size, self.njoint, 2], name='2d_gt')
        self.pl_intrin_mat = tf.placeholder(tf.float32, shape=[self.batch_size*self.window_size, 3, 2], name='intrin_mat')
        self.pl_estimate_valid = tf.placeholder(tf.float32, shape=[self.batch_size*self.window_size],name='current_valid')
        self.pl_seg = tf.placeholder(tf.float32,
                                     shape=[self.batch_size * self.window_size, self.setting.height, self.setting.width],
                                     name='segmentation')
        self.pl_dmap = tf.placeholder(tf.float32,
                                     shape=[self.batch_size * self.window_size, self.setting.height, self.setting.width],
                                     name='depth_maps')



        self.data_prep(self.pl_dmap, self.pl_2d_gt_pose)
        self.simplified_points = self.build_sampler(self.pcloud, self.simplified_num)

        _temperature = tf.get_variable("temperature",
                                       initializer=tf.constant(self.setting.initial_temperature, dtype=tf.float32),
                                       trainable=True, dtype=tf.float32, )
        self._temperature_safe = tf.maximum(_temperature, 0.1)
        self.sigma = self._temperature_safe ** 2


        

        self.tmp_simplified_points = tf.reshape(self.simplified_points,
                                           [self.batch_size * self.window_size, self.njoint * self.simplified_num, 3])
        tmp_pcloud = tf.reshape(self.pcloud,
                                [self.batch_size * self.window_size, self.njoint * self.setting.sample_pixel_total, 3])
        
        idx = tf.where(self.pl_estimate_valid)
        valid_sim = tf.gather_nd(self.simplified_points, idx)
        self.patch_pc = tf.reshape(valid_sim,
                              [-1, self.simplified_num, 3])
        
        self.G_offset = self.generator(self.tmp_simplified_points)
        self.G_pose = self.G_offset + self.pose_3d_init


        bbox_size = tf.tile(self.bbox_size_tf, [self.njoint, 1])
        self.pred_3d = tf.add(tf.multiply(self.G_pose, bbox_size), self.bbox_center)

        pred_z = self.pred_3d[:,:,2]
        pred_z = tf.tile(tf.expand_dims(pred_z,2),[1,1,3])
        self.projection_2d = tf.matmul(self.pred_3d/pred_z, self.pl_intrin_mat)

        # build loss
        self._get_loss()
        with tf.variable_scope("", reuse=True):
            self.W = tf.get_variable('encoder_conv_layer_0/W')
        self.grad_check = tf.gradients(self.density_loss, self.W)

        all_vars = tf.trainable_variables()
        not_z_vars = [var for var in all_vars if not var.name.startswith('znet')]
        print("trained variables:")
        for var in all_vars:
            print(var.name)
        self.g_lr = tf.train.exponential_decay(self.g_lr, self.global_step, 100, 0.95, staircase=True)
        self.opt_g = tf.train.AdamOptimizer(self.g_lr).minimize(self.loss_total, var_list=not_z_vars)

        # validation

        # save model and scalar
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        
        self.saver = tf.train.Saver(max_to_keep=1)
        if not self.load_model:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load ckpt:",ckpt.model_checkpoint_path)
    
        self.setup_summaries()



    def setup_summaries(self):
        '''

        :param loss:
        :return:
        '''

        scalar_array = [tf.summary.scalar('loss/loss_total', self.loss_total),
                        tf.summary.scalar('loss/reconstruction', self.loss_pose),
                        tf.summary.scalar('loss/self_consistance', self.loss_g_time),
                        tf.summary.scalar('loss/loss_2d', self.loss_pose_2d),
                        tf.summary.scalar('loss/simplified_loss', self.loss_simp),
                        # tf.summary.scalar('loss/loss_weight', self.loss_weight),
                        tf.summary.scalar('loss/loss_projection', self.loss_proj),
                        tf.summary.scalar('loss/loss_density', self.density_loss),
                        tf.summary.scalar('loss/loss_pc_3d', self.loss_pc_3d)

                       ]


        self.summary_op_all = tf.summary.merge(scalar_array)

    def _get_loss(self):
        # simplification loss
        self.loss_simp, dist, idx, dist2 = self._get_simplification_loss(self.simplified_points,
                                                                         self.pcloud,
                                                                         self.simplified_num)

        # projection loss
        self.loss_proj = self.sigma

        self.loss_bone = self.get_bone_length_loss(self.pred_3d, self.setting)

        self.density_loss = self.get_density_loss(self.setting.radius, self.setting.M, self.patch_pc)
       

        self.loss_pc_3d = self._get_simplified_3d()

        # pose regression loss
        self.loss_pose = tf.div(tf.reduce_sum(tf.multiply(tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.pred_3d, self.pl_3d_gt_pose)), axis=2)), axis=1),
                                                          self.pl_estimate_valid)),
                                (tf.reduce_sum(self.pl_estimate_valid) + 0.01))
        self.loss_pose_2d = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.projection_2d, self.pl_2d_gt_pose)), axis=2)))
        
        if self.window_size != 1:
            self.loss_g_time = self._get_self_consistance_loss(self.pred_3d)
            self.loss_regression = self.loss_pose  + 0.0001 * self.loss_g_time #+ self.loss_pose_2d * 0.01 #+ self.extra_loss
        else:
            self.loss_regression = self.loss_pose_2d #+ self.extra_loss

        self.loss_total = self.loss_regression + \
                          100*self.loss_simp + \
                          self.loss_proj + \
                          self.loss_bone + \
                          self.density_loss #+ \
                          # self.loss_pc_3d


    def _get_self_consistance_loss(self, pred_poses):
        shape_d = int(pred_poses.get_shape().as_list()[0]/ self.batch_size)
        pred_poses = tf.reshape(pred_poses, [self.batch_size,shape_d,self.njoint,3])
        
        v_loss, a_loss = self._selfloss(pred_poses)
        v_tmp = tf.norm(v_loss,ord=2,axis=3)
        a_tmp = tf.norm(a_loss,ord=2,axis=3)
        return tf.reduce_mean(v_tmp) + tf.reduce_mean(a_tmp)

    def get_bone_length_loss(self, pr, setting):
        # pr are denormalized poses(pred)

        pr_bone = []
        gt_bone = []
        skeleton_id = setting.skeleton_id

        for i in range(len(skeleton_id)):
            joint_id = skeleton_id[i]
            pr_bone_tmp = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(pr[:, joint_id[0], :], pr[:, joint_id[1], :]), 2), axis=1))
            pr_bone.append(pr_bone_tmp)
            gt_bone.append(tf.constant(setting.skeleton[i], dtype=tf.float32))

        loss = None
        mean_ratio = 0
        ratio = []
        for i in range(len(pr_bone)):
            temp_ratio = tf.div(pr_bone[i], gt_bone[i])
            ratio.append(temp_ratio)
            mean_ratio = mean_ratio + temp_ratio
        mean_ratio = mean_ratio / len(pr_bone)
        for i in range(len(pr_bone)):
            if loss is None:
                loss = tf.square(tf.subtract(ratio[i], mean_ratio))
            else:
                loss = loss + tf.square(tf.subtract(ratio[i], mean_ratio))
            loss = tf.reduce_sum(loss) / len(skeleton_id)
            return loss

    def _selfloss(self, pre_poses):
        """
        :param pre_poses: previous estimated poses (over 3) (batch_size, time_step, njoint,3)
        :return: mean volecity and mean acceleration of previous poses (joint_num*3)
        """
        if pre_poses.get_shape().as_list()[1] < 3:
            return tf.Variable(0)
        framerate = 1 / 30
        pre_num = pre_poses.get_shape().as_list()[1]
       
        v_tmp = []
        a_tmp = []
        v_loss = []
        a_loss = []
        for j in range(1,pre_num):
            v_tmp.append((pre_poses[:, j, :, :] - pre_poses[:,j-1, :, :]) / framerate) #(pred_num-1,batch_size, njoint,3)
        for j in range(1,pre_num-1):
            x1 = pre_poses[:, j , :, :] - pre_poses[:, j-1, :, :]
            x2 = pre_poses[:, j + 1, :, :] - pre_poses[:, j , :, :]
            del_x = x2 - x1  # (batch_size, njoint,3)
            a = del_x / (framerate * framerate)
            a_tmp.append(a) #(pred_num-2,batch_size, njoint,3)
        v_tmp = tf.stack(v_tmp, axis=1)
        if pre_num > 3:
            a_tmp = tf.stack(a_tmp, axis = 1)
        else:
            a_tmp = tf.expand_dims(a_tmp,axis=1)
        for i in range(1, v_tmp.get_shape().as_list()[1]):
            v_loss.append(v_tmp[:,i,:,:] - v_tmp[:,i-1,:,:])  #([],B,njoint,3)
        if pre_num > 3:
            for i in range(1,a_tmp.get_shape().as_list()[1]):
                a_loss.append(a_tmp[:,i,:,:] - a_tmp[:,i-1,:,:])  #([],B,njoint,3 )
        else:
            a_loss = tf.zeros_like(a_tmp[:,0,:,:])
        v_loss = tf.stack(v_loss,axis=1)
        if pre_num >3:
            a_loss = tf.stack(a_loss,axis=1)
        else:
            a_loss = tf.expand_dims(a_loss, axis=1)
        return v_loss, a_loss

    def _get_simplification_loss(self, ref_pc, samp_pc, pc_size):
        ref_pc = tf.reshape(ref_pc, [self.batch_size*self.window_size*self.njoint, self.simplified_num, 3])
        samp_pc = tf.reshape(samp_pc, [self.batch_size*self.window_size*self.njoint, self.nsampled, 3])
        cost_p1_p2, idx, cost_p2_p1, _ = nn_distance(samp_pc, ref_pc)
        dist = cost_p1_p2
        dist2 = cost_p2_p1
        max_cost = tf.reduce_max(cost_p1_p2, axis=1)
        max_cost = tf.reduce_mean(max_cost)

        self.nn_distance = tf.reduce_mean(
            cost_p1_p2, axis=1, keep_dims=True
        ) + tf.reduce_mean(cost_p2_p1, axis=1, keep_dims=True)

        cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
        cost_p2_p1 = tf.reduce_mean(cost_p2_p1)

        w = pc_size / 64.0
        if self.is_denoising:
            loss = cost_p1_p2 + max_cost + 2 * w * cost_p2_p1
        else:
            loss = cost_p1_p2 + max_cost + w * cost_p2_p1

        return loss, dist, idx, dist2

    def get_density_loss(self, eps, M, sample_pc):
        
        idx, pts_cnt = query_ball_point(eps, M, sample_pc, sample_pc)  # idx shape [B,np,ns]
        M_pad = tf.ones(shape=[tf.shape(sample_pc)[0], sample_pc.get_shape()[1]], dtype=tf.int32) * M
        not_core = tf.cast(tf.not_equal(pts_cnt, M_pad),
                           tf.float32)  # core point is 0, not core point is 1 shape=[B, np]
        core = tf.cast(tf.equal(pts_cnt, M_pad),
                           tf.float32)  # core point is 0, not core point is 1 shape=[B, np]
        core_sum = tf.reduce_sum(core, axis=1)
        not_cor_idx = tf.where(not_core)  # shape = (?, 2)
        core_idx = tf.where(core)
        batch_idx = not_cor_idx[:, 0:1]

        batch_idx = tf.tile(batch_idx, [1, M])
        batch_idx = tf.reshape(batch_idx, [-1])
        neighbor = tf.gather_nd(idx, not_cor_idx)  # (?, M)
        neighbor = tf.cast(tf.reshape(neighbor, [-1]), tf.int64)
        new_idx = tf.concat([batch_idx[:, tf.newaxis], neighbor[:, tf.newaxis]], axis=1)
        data = tf.gather_nd(not_core, new_idx)  # [?*M]
        data = tf.cast(tf.reshape(data, [-1, M]), tf.float32)  # [?,M]
        data = tf.reduce_prod(data, axis=1) #[?]
        noise_idx = tf.where(data)
        noise_coor = tf.gather_nd(not_cor_idx, noise_idx) #[noise_num, 2]
        noise_batch = noise_coor[:, 0:1]  # nn*1
        ones = tf.ones_like(not_core)
        nine = tf.ones_like(not_core) * 999
        not_core_dis = tf.cast(tf.where(tf.not_equal(pts_cnt, M_pad), nine, ones), tf.float32)
        not_core_indice = tf.gather_nd(not_core_dis, noise_batch)  # nn * np if core 1, not core 999
        all_points = tf.gather_nd(sample_pc, noise_batch) * \
                     tf.tile(not_core_indice[..., tf.newaxis], [1, 1, 3]) # nn * np *3
        noise_points = tf.gather_nd(sample_pc, noise_coor)  # nn * 3

        val, nearest_idx = knn_point(1, all_points, noise_points[:, tf.newaxis, :])
        nearest_core = group_point(all_points, nearest_idx) # [nn, 1, 1, 3]
        nearest_core = tf.squeeze(nearest_core)
        error = tf.norm(nearest_core - noise_points, axis=1)

        # loss = tf.reduce_sum(noise_points ** 2) / (tf.reduce_sum(self.pl_estimate_valid) *
        #                                                       self.njoint * self.simplified_num + 0.01)
        loss = tf.reduce_sum(error) / (tf.reduce_sum(self.pl_estimate_valid) *
                                                   self.njoint * self.simplified_num + 0.01)

        # use knn calculate no core points
        

        return loss

    def _get_simplified_3d(self):
        perjoint_pc = tf.reshape(self.tmp_simplified_points, [-1, self.njoint, self.simplified_num, 3])
        residuce = perjoint_pc - tf.tile(self.norm_gt_3d[..., tf.newaxis, :], (1, 1, self.simplified_num, 1))
        error = tf.norm(residuce, axis=-1)
        mean_err = tf.reduce_mean(error, axis=[1, 2])
        return tf.div(tf.reduce_sum(tf.multiply(mean_err, self.pl_estimate_valid)),
                      tf.reduce_sum(self.pl_estimate_valid)+0.01)

    def data_prep(self, dmap, joint_2d):
        tmp_a = tf.expand_dims(tf.clip_by_value(joint_2d[:, :, 0],
                                                0 + self.setting.sample_pixel, 319 - self.setting.sample_pixel),
                               axis=2)
        tmp_b = tf.expand_dims(tf.clip_by_value(joint_2d[:, :, 1],
                                                0 + self.setting.sample_pixel, 239 - self.setting.sample_pixel),
                               axis=2)

        bbox_set = tf_util.GetBatch2DBBoxSet_tf(tf.concat([tmp_a, tmp_b], axis=2), self.setting)
        pcloud, self.input_pc_weights = tf_util.GetEveryPcloud_tf(dmap,
                                                                  self.pl_seg,
                                                                  bbox_set,
                                                                  self.setting,
                                                                  is_demo='ITOP')
        self.ori_pc = pcloud

        z_a = tf.expand_dims(tf.clip_by_value(joint_2d[:, 8, 0],
                                              0 + self.setting.zsample_pixel, 319 - self.setting.zsample_pixel),
                             axis=1)
        z_b = tf.expand_dims(tf.clip_by_value(joint_2d[:, 8, 1],
                                              0 + self.setting.zsample_pixel, 239 - self.setting.zsample_pixel),
                             axis=1)
        root_box = tf_util.Getcorebox_tf(tf.concat([z_a, z_b], axis=1), self.setting)
        rootpc = tf_util.GetRootPcloud_tf(dmap, root_box, self.setting, is_demo='ITOP')
        Npoint = int(self.setting.zsamples)
        mean = tf.tile(tf.expand_dims(tf.reduce_mean(rootpc, axis=1), axis=1), [1, Npoint, 1])

        # normalize root joint's pointcloud
        rootpc = rootpc - mean
        z_tf = self.z_predictor(rootpc, is_training=self.pl_is_training)

        # denormalize pred Z value
        z_tf = z_tf + mean[:, 0, 2:]
        self.ori_z = z_tf
        mean_z = tf.tile(z_tf, [1, self.setting.joint_num])

        pose_3d_init = tf_util.Imgcoor2Pcloud_tf(dmap, tf.concat([tmp_a, tmp_b], axis=2), is_demo='ITOP', init_z=mean_z)
        self.ori_ini = pose_3d_init
        center = pose_3d_init[:, 8, :]

        self.bbox_size_tf = tf.cast(tf.reshape(tf.constant(self.setting.bbox_size), [1, -1]), tf.float32)
        bbox = tf.concat([center, tf.tile(self.bbox_size_tf, [self.batch_size * self.setting.window_size, 1])], axis=1)
        bbox = tf.tile(tf.expand_dims(bbox, axis=1), [1, self.njoint, 1])
        self.bbox_center = bbox[..., :3]

        # Point cloud normalization
        self.pcloud, _ = tf_util.PcloudNomalization_tf(pcloud, bbox, self.setting.sample_pixel_total)
        pose_3d_init = tf.expand_dims(pose_3d_init, axis=2)
        pose_3d_init, _ = tf_util.PcloudNomalization_tf(pose_3d_init, bbox, 1)
        self.pose_3d_init = tf.squeeze(pose_3d_init, axis=2)
        world_coor = tf.expand_dims(self.pl_3d_gt_pose, axis=2)
        world_coor, bbox = tf_util.PcloudNomalization_tf(world_coor, bbox, 1)
        self.norm_gt_3d = tf.squeeze(world_coor, axis=2)

    def build_sampler(self, input_pc, pc_dim_out, non_linearity=tf.nn.relu):
        input_pc = tf.reshape(input_pc, [self.batch_size * self.window_size * self.njoint, self.nsampled, 3])
        encoder_args = {
            "n_filters": [64, 128, 256, 512, 256],
            "filter_sizes": [1],
            "strides": [1],
            "non_linearity": non_linearity,
            "b_norm": False,
            "verbose": True,
        }
        layer = encoder_with_convs_and_symmetry(input_pc, **encoder_args)

        decoder_args = {
            "layer_sizes": [256, 512, np.prod([pc_dim_out, 3])],
            "b_norm": False,
            "b_norm_finish": False,
            "verbose": True,
        }
        out_signal = decoder_with_fc_only(layer, **decoder_args)
        simplified = tf.reshape(out_signal, [-1, self.njoint, pc_dim_out, 3])
        # weights = tf.reshape(weights, [-1, self.njoint, self.setting.sample_pixel_total])
        return simplified

    def build_projector(self, point_cloud, query_cloud, sigma, group_size, hard,  pred_weights, mode):
        # grouped_points: (batch_size, num_out_points, group_size, 3)
        if mode == 'test':
            group_size = 1
        # point_cloud has shape [BW, JT, 3], query_cloud [BW, JS, 3]
        grouped_points, grouped_weights = _group_points_weights(point_cloud, query_cloud, pred_weights,
                                                              group_size)

        # grouped_weights have shape [BW, JS, group_size, 1], these weights will calculate distance later
        # predict weight
        dist = _get_distances_weights(grouped_points, query_cloud, group_size, sigma, grouped_weights)

        # pass through softmax to get weights
        weights = tf.nn.softmax(-dist, axis=2)
        if hard:
            # convert softmax weights to one_hot encoding
            weights = tf.one_hot(tf.argmax(weights, axis=2), depth=self._group_size)
            weights = tf.transpose(weights, perm=[0, 1, 3, 2])

        # get weighted average of grouped_points
        projected_point_cloud = tf.reduce_sum(
            grouped_points * weights, axis=2
        )  # (batch_size, num_out_points, 3)
        
        return projected_point_cloud


    def built_weight_predictor(self, point_cloud, bn_decay=None, points=None, name='weight', reuse=tf.AUTO_REUSE):
        njoint = self.njoint
        batch_size = self.batch_size
        end_points = {}

        with tf.variable_scope(name, reuse=reuse):

            l0_xyz = point_cloud
            if points is not None:
                l0_points = points
            else:
                l0_points = None
            end_points['l0_xyz'] = l0_xyz

            l1_xyz, l1_points, l1_indices = pointnet_sa_module_multi_joints(l0_xyz, l0_points, npoint=6, radius=0.05,
                                                                            nsample=4, njoint=njoint, mlp=[32, 32, 64],
                                                                            mlp2=None, group_all=False,
                                                                            bn_decay=bn_decay,
                                                                            scope='layer1')
            net = tf.reshape(l1_points, [self.batch_size * self.window_size, -1])
            temp = fully_connected(net, self.simplified_num * self.setting.group_size * 2, 'fc')
            temp = tf.reshape(temp, shape=[self.batch_size*self.window_size, -1, 2])
            weight, _ = tf.split(tf.nn.softmax(temp), [1, 1], 2)

        return weight

    def generator(self, point_cloud, bn_decay=None, points=None, name='gen', reuse=tf.AUTO_REUSE):
        """

        :param point_cloud: (Batch_size * time_len, njoint, sample_num,3)
        :param bn_decay:
        :param points:
        :param name:
        :param reuse:
        :return:
        """
        njoint = self.njoint
        batch_size = self.batch_size
        state_size = 256

        with tf.variable_scope(name, reuse=reuse):
            features = pointnet_sa_new(point_cloud, [64, 128, 1024], bn_decay=bn_decay, scope=name, bn=False,
                                       use_nchw=False)

            net = tf.reshape(features, [self.batch_size * self.window_size, -1])
            net = tf.concat([net, tf.reshape(self.pose_3d_init, [self.batch_size * self.window_size, -1])], axis=1)

            # use LSTM layers replace the fully connection layers of the PointNet++ Network
            # reshape the feature "net" into [batch_size, time_step, -1]
            net = tf.reshape(net, [batch_size, self.window_size, -1])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size, state_is_tuple=True)
            initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, net, initial_state=initial_state)

            # get final joints' position by output(batch_size, time_step, state_size)
            outputs = tf.reshape(outputs, [-1, state_size])

            poses = tf_util.fully_connected(outputs, njoint * 3, 'fc', activation_fn=None)
            poses = tf.reshape(poses, [batch_size * self.window_size, njoint, 3])

        return poses

    def z_predictor(self, point_cloud, is_training, name='znet', reuse=tf.AUTO_REUSE):
        N = point_cloud.shape[1]
        with tf.variable_scope(name, reuse=reuse):
            net = tf_util.conv1d(point_cloud, 64, 3, scope='conv1')
            tmp = net
            net = tf_util.conv1d(net, 64, 3, scope='conv2', activation_fn=None)
            net = tf.nn.relu(tmp + net)
            net = tf_util.conv1d(net, 128, 3, scope='conv3')
            tmp = net
            net = tf_util.conv1d(net, 128, 3, scope='conv4', activation_fn=None)
            net = tf.nn.relu(net + tmp)
            net = tf_util.conv1d(net, 256, 3, scope='conv5')
            tmp = net
            net = tf_util.conv1d(net, 256, 3, scope='conv6', activation_fn=None)
            net = tf.nn.relu(net + tmp)
            net = tf_util.conv1d(net, 512, 3, scope='conv7')
            tmp = net
            net = tf_util.conv1d(net, 512, 3, scope='conv8', activation_fn=None)
            net = tf.layers.average_pooling1d(net, pool_size=int(N), padding="SAME", strides=int(N))
            net = tf.squeeze(net, axis=1)
            net = tf_util.fully_connected(net, num_outputs=1, bn=False, is_training=is_training, scope='tfc1',activation_fn=None)
            net = tf.reshape(net, [-1, 1])
        return net

    def train(self):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        print('PID: ', os.getpid())
        iters = self.MAXITER
        for iter in range(int(iters)):
            self.global_step = iter
            # load the training data
            tr_dmap, tr_image_coor, tr_world_coor, \
            tr_cur_valid, _, tr_seg = self.train_set.next_normalized_batch(
                self.batch_size, self.setting)

            feed_dict = {
                         self.pl_seg: tr_seg,
                         self.pl_3d_gt_pose: tr_world_coor,
                         self.pl_dmap: tr_dmap,
                         self.pl_2d_gt_pose: tr_image_coor,
                         self.pl_intrin_mat: self.setting.rotmat,
                         self.pl_is_training: True,
                         self.pl_estimate_valid: tr_cur_valid,
                         }
            g_opt = [self.opt_g, self.loss_total, self.loss_pose, self.loss_pose_2d,
                     self.loss_g_time, self.density_loss, self.grad_check]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                
                _, loss_total, loss_pose, loss_pose_2d, loss_g_time, loss_density, \
                gradients = self.sess.run(g_opt, feed_dict=feed_dict)
            if iter == 0:
                print("the density related gradients ", gradients)
            if iter % 10 == 0:
                summary = self.sess.run(self.summary_op_all, feed_dict=feed_dict)
                if loss_pose != 0:
                    self.train_summary_writer.add_summary(summary, global_step=iter)
                print('Iteration: %d; 3d pose loss: %g, density loss: %g' % (iter, loss_pose,
                                                                                        loss_density))

            if iter % 100 == 0:
                ts_dmap, ts_image_coor, ts_world_coor, \
                ts_cur_valid, _, ts_seg = self.val_set.next_normalized_batch(
                    self.batch_size, self.setting)
                feed_dict = {
                             self.pl_seg: ts_seg,
                             self.pl_3d_gt_pose: ts_world_coor,
                             self.pl_dmap: ts_dmap,
                             self.pl_2d_gt_pose: ts_image_coor,
                             self.pl_intrin_mat: self.setting.rotmat,
                             self.pl_is_training: False,
                             self.pl_estimate_valid: ts_cur_valid,
                             }
                summary, pred_pose, loss_total, loss_g_time, pred_2d, \
                loss_pose, ts_init_pose, ts_pcloud, density_loss,\
                    ori_pc, ori_z, ts_sample_pc = self.sess.run([self.summary_op_all,
                                                                         self.G_pose, self.loss_total,
                                                                         self.loss_g_time,
                                                                         self.projection_2d,
                                                                        self.loss_pose,
                                                                        # self.simplified_points,
                                                                        self.pose_3d_init,
                                                                        self.pcloud,
                                                                        # self.projected_point_cloud,
                                                                        self.density_loss,
                                                                        self.ori_pc,
                                                                        self.ori_z,
                                                                        self.tmp_simplified_points
                                                                        ],
                                                                        feed_dict=feed_dict)

                if loss_pose != 0:
                    self.valid_summary_writer.add_summary(summary, global_step=iter)
                print('%s ---> Pose 3d loss: %g, density loss: %g' % (datetime.datetime.now(),
                                                                                 loss_pose,
                                                                                 density_loss
                                                                                        ))
                
            if iter % 1000 == 0:
                self.saver.save(self.sess, self.save_path + 'model.ckpt', iter)
                io.savemat('result.mat', {'depth_map': ts_dmap,
                                                         'joint_3d_gt': ts_world_coor,
                                                         'joint_2d_gt': ts_image_coor,
                                                         'joint_3d_pred': pred_pose,
                                                         'joint_3d_initial': ts_init_pose,
                                                         'pcloud': ts_pcloud,
                                                         'cur_valid': ts_cur_valid,
                                                        # 'sample_points':ts_samples,
                                                        # 'proj_pc': proj_pc
                                                        'ori_pc': ori_pc,
                                                        'ori_z': ori_z,
                                                        'sample_points':ts_sample_pc,
                                                        'segmentation': ts_seg
                                                      })


    def test(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        print('PID: ', os.getpid())

        epoch = self.val_set.epochs_completed
        id_array = []
        pckh_array = []
        mean_array = []
        total_cont = []

        all_pred = []
        all_gt = []
        all_valid = []
        all_color = []
        all_depth = []
        all_oripc = []
        all_samplepc = []
        allestpck_count = 0

        allestpck = np.zeros(6)

        tic()
        # for ep in range(100):
        while self.val_set.epochs_completed <= epoch:
            ts_dmap, ts_image_coor, ts_world_coor, \
            ts_cur_valid, ts_id, ts_seg = self.val_set.next_normalized_batch(
                self.batch_size, self.setting)

            feed_dict = {
                             self.pl_seg: ts_seg,
                             self.pl_3d_gt_pose: ts_world_coor,
                             self.pl_dmap: ts_dmap,
                             self.pl_2d_gt_pose: ts_image_coor,
                             self.pl_intrin_mat: self.setting.rotmat,
                             self.pl_is_training: False,
                             self.pl_estimate_valid: ts_cur_valid,
                         }
            pred_2d, init_3d_pose, denorm_pred, loss_g, ts_pcloud, ts_bbox, ori_ini, sample_points = self.sess.run([self.projection_2d, self.pose_3d_init,
                                                                                         self.pred_3d,
                                                                                         self.loss_total,

                                                                                         self.pcloud,
                                                                                         # self.simplified_points,
                                                                                         self.bbox_center,
                                                                                         self.ori_ini,
                                                                                         self.tmp_simplified_points
                                                                                     ],
                                                                                feed_dict=feed_dict)
            all_pred.append(denorm_pred)
            all_gt.append(ts_world_coor)
            all_valid.append(ts_cur_valid)
            all_depth.append(ts_dmap)
            all_oripc.append(ts_pcloud)
            all_samplepc.append(sample_points)
            id_array.append(ts_id)
            color = np.zeros([ts_dmap.shape[0], 240, 320, 3], dtype=np.uint8)
            # for i in range(ts_dmap.shape[0]):
            #     color[i] = utils.depth_map_to_image(ts_dmap[i])
            # cv2.imshow("title", color[0].copy())
            # cv2.waitKey(0)
            all_color.append(color)

            # exit()
            #plt_img = utils.depth_map_to_image(ts_dmap[0], joints=pred_2d[0], joints_true=pred_2d[0])
            # cv2.imshow("title",plt_img)
            # cv2.waitKey(0)
            # pckh = utils.calPCKH(tmp_pose, tmp_gt)


            for k in [5, 10, 15, 20, 25, 30]:
                pckh, cont, = utils.calPck_time(denorm_pred, ts_world_coor, ts_cur_valid, threshold=k)
                if k == 5:
                    allestpck_count += np.sum(cont)

                allestpck[int(k/5-1)] += np.sum(cont)*np.mean(pckh)



            pckh, cont = utils.calPck_time(denorm_pred,
                                                              ts_world_coor,
                                                              ts_cur_valid,
                                                              threshold=10)

            if np.sum(cont) != 0 :
                
                cont = np.expand_dims(cont, 0)
                mean0 = np.sum(np.matmul(cont, np.linalg.norm(denorm_pred - ts_world_coor, ord=2, axis=2)))/(np.sum(cont)*self.njoint)
                mean = np.sum(np.matmul(cont, np.linalg.norm(denorm_pred - ts_world_coor, ord=2, axis=2)), axis=0)/np.sum(cont)
                print("PCKH: %g, Mean joint error: %g" % (np.mean(pckh), mean0))
                pckh_array.append(pckh)
                mean_array.append(mean)
                total_cont.append(np.sum(cont))
        

        toc()

        id_array = np.array(id_array)
        pckh_np = np.array(pckh_array)
        mean_np = np.array(mean_array)
        total_cont = np.expand_dims(np.array(total_cont),0)
        print("---> PCKH: %g, Mean joint error: %g" %
              (np.mean(np.matmul(total_cont,pckh_np)/np.sum(total_cont)),
               np.mean(np.matmul(total_cont,mean_np)/np.sum(total_cont))))
        print(np.matmul(total_cont,mean_np)/np.sum(total_cont))
        print("---> per joint pckh \n %s" % str(np.matmul(total_cont,pckh_np)/np.sum(total_cont)))
        print(np.sum(total_cont))
        print("********************************************************")
        for i in [5,10,15,20,25,30]:
            print("***********pck with threshold = %d**************"% i)
            print("estimation pck = %g" % (allestpck[int(i/5-1)]/allestpck_count))

        io.savemat('test.mat', {#'depth_map': np.concatenate(all_depth, 0),
                                                                  'joint_3d_gt': np.concatenate(all_gt, 0),
                                                                  'human_id': np.concatenate(id_array, 0),
                                                                  'joint_3d_pred': np.concatenate(all_pred, 0),
                                                                  #'pcloud': np.concatenate(all_oripc, 0),
                                                                  'cur_valid': np.concatenate(all_valid, 0),
                                                                  #'proj_pc': np.concatenate(all_samplepc, 0),
                                                                  })


    def demo(self):
        print('PID: ', os.getpid())

        rot_mat = np.zeros((self.batch_size*self.setting.window_size, 3, 3), dtype=np.float32)
        pred_all = []
        init_poses = []
        poses_2d = []
        while self.demo_set.epochs_completed < 1:

            ts_pcloud, ts_init_pose, ts_bbox, ts_2d = \
                self.demo_set.next_demo_norm_batch(self.batch_size,self.setting)

            for j in range(self.batch_size*self.window_size):
                rot_mat[j, ...] = utils.axang2rotm([0, 1, 0], 0)

            center = ts_bbox[..., 0:3]

            feed_dict = {self.pl_sampled_points: ts_pcloud,
                         self.pl_3d_init_pose: ts_init_pose,
                         self.pl_intrin_mat: self.setting.demomat,
                         self.pl_rot_mat: rot_mat,
                         self.pl_bbox_center: center
                         }

            pred_pose = self.sess.run(self.G_pose, feed_dict=feed_dict)
            pred_np = utils.DeNormalization(pred_pose, ts_bbox)
            pred_all.append(pred_np)
            init_poses.append(ts_init_pose)
            poses_2d.append(ts_2d)
        save_pred = np.array(pred_all).reshape([-1,self.setting.joint_num,3])
        save_initial = np.array(init_poses).reshape([-1,self.setting.joint_num,3])
        save_joint_2d = np.array(poses_2d).reshape([-1,self.setting.joint_num,2])
        io.savemat("demo_result.mat", {"pred_poses": save_pred, "initial_poses":save_initial,
                                       "poses_2d": save_joint_2d})



    def test_selected_output(self, index):
        print('PID: ', os.getpid())
        rot_mat = np.zeros((self.batch_size * self.window_size, 3, 3), dtype=np.float32)
        gt_poses = []
        pred_poses = []
        pcloud = []
        dmap = []
        samplepc = []

        for i in range(int((index[-1]-index[0])/self.window_size)):
            ts_dmap, ts_image_coor, ts_world_coor, \
            ts_cur_valid, ts_seg = self.val_set.next_selected_batch(
                index[i*self.window_size:i*self.window_size+self.window_size], self.setting)

            feed_dict = {
                self.pl_seg: ts_seg,
                self.pl_3d_gt_pose: ts_world_coor,
                self.pl_dmap: ts_dmap,
                self.pl_2d_gt_pose: ts_image_coor,
                self.pl_intrin_mat: self.setting.rotmat,
                self.pl_is_training: False,
                self.pl_estimate_valid: ts_cur_valid,
            }
            pred_2d, init_3d_pose, denorm_pred, ts_pcloud, sample_points = self.sess.run(
                [self.projection_2d, self.pose_3d_init,
                 self.pred_3d,
                 self.pcloud,
                 self.tmp_simplified_points
                 ],
                feed_dict=feed_dict)
            gt_poses.append(ts_world_coor)
            pred_poses.append(denorm_pred)
            pcloud.append(ts_pcloud)
            dmap.append(ts_dmap)
            samplepc.append(sample_points)

        gt_poses = np.concatenate(gt_poses, 0)
        pred_poses = np.concatenate(pred_poses, 0)
        pcloud = np.concatenate(pcloud, 0)
        dmap = np.concatenate(dmap, 0)
        samplepc = np.concatenate(samplepc, 0)
        print(gt_poses.shape, pred_poses.shape, pcloud.shape, samplepc.shape)
        io.savemat('pick_results.mat', {"dmap": dmap,
                                                        "gt_poses": gt_poses,
                                                       "pred_poses": pred_poses,
                                                       "samplepc": samplepc})
        print('Output Complete!')

    def performance_test(self):
        all_time = []
        for num in range(1000):
            ts_dmap, ts_image_coor, ts_world_coor, \
            ts_cur_valid, _, ts_seg = self.val_set.next_normalized_batch(
                self.batch_size, self.setting)

            t1 = time.time()
            feed_dict = {
                self.pl_seg: ts_seg,
                self.pl_3d_gt_pose: ts_world_coor,
                self.pl_dmap: ts_dmap,
                self.pl_2d_gt_pose: ts_image_coor,
                self.pl_intrin_mat: self.setting.rotmat,
                self.pl_is_training: False,
                self.pl_estimate_valid: ts_cur_valid,
            }
            denorm_pred = self.sess.run(self.pred_3d,
                feed_dict=feed_dict)
            t2 = time.time()
            print(t2-t1)
            if num != 0:
                all_time.append(t2-t1)
        print("mean time %g" % np.mean(all_time))

    def seg_generation(self):
        dmap_list = []
        seg_list = []
        valid_list = []
        for _ in range(100):
            ts_dmap, ts_image_coor, ts_world_coor, \
            ts_cur_valid, _, ts_seg = self.val_set.next_normalized_batch(
                self.batch_size, self.setting)
            dmap_list.append(ts_dmap)
            seg_list.append(ts_seg)
            valid_list.append(ts_cur_valid)
        dmap_list = np.concatenate(dmap_list, 0)
        seg_list = np.concatenate(seg_list, 0)
        valid_list = np.concatenate(valid_list, 0)
        io.savemat('seg_ex.mat', {"dmap":dmap_list, "seg":seg_list,"valid":valid_list})
