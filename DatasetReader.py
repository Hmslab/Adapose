import h5py
import utils.np_utils as utils
import numpy as np
from scipy import io
import sys
import cv2
import tensorflow as tf
# from utils import openpose_util as op
from scipy.ndimage.interpolation import zoom
import os
import copy

sys.setrecursionlimit(90000)


def loademodata(depth_dir, color_dir):
    depth_files = [file for file in os.listdir(depth_dir) if file[-3:] == 'png']
    depth_files = [os.path.join(depth_dir, file) for file in depth_files]
    color_files = [file for file in os.listdir(color_dir) if file[-3:] == 'png']
    color_files = [os.path.join(color_dir, file) for file in color_files]
    depth_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    color_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    assert len(color_files) == len(depth_files)
    depth = np.zeros([len(color_files), 240, 320], dtype=np.float32)
    color = np.zeros([len(color_files), 240, 320, 3], dtype=np.uint8)
    for num in range(len(depth_files)):
        depth_tmp = cv2.imread(depth_files[num], -1) / 1000
        color_tmp = cv2.imread(color_files[num])
        depth[num] = zoom(depth_tmp, (240/depth_tmp.shape[0], 320/depth_tmp.shape[1]), order=1)
        color[num] = cv2.resize(color_tmp, (320, 240))
    return depth, color

def creat_joint2d_noise(joint_2d):
    mask = np.ones([joint_2d.shape[0], joint_2d.shape[1]])
    randata = np.random.rand(joint_2d.shape[0], 6)
    njoint = np.arange(15)
    njoint = np.setdiff1d(njoint,[8])
    njoint = np.tile(njoint[np.newaxis,:],[joint_2d.shape[0], 1])
    njoint = np.transpose(njoint)
    np.random.shuffle(njoint)
    njoint = np.transpose(njoint)
    index = np.where(randata < 0.2)
    for i in range(len(index[0])):
        mask[index[0][i], njoint[index[0][i], index[1][i]]] = 0
    return mask

class DatasetReader:
    dataset_length = 0
    batch_offset = 0
    epochs_completed = 0
    select_index = -1

    # data in the dataset files
    perm = None
    depth_maps = None
    image_coordinates = None
    real_world_coordinates = None
    visible_joints = None
    is_valid = None

    def __init__(self, path, setting, is_shuffle=True, is_demo=False):
        
        depth_files = h5py.File(path + 'depth_map.h5', 'r')
        labels = h5py.File(path + 'labels.h5', 'r')

        self.window_size = setting.window_size
        self.depth_maps = depth_files['data']
        self.segmentation = labels['segmentation']
        self.dataset_length = self.depth_maps.maxshape[0]
        self.is_valid = labels['is_valid']
        self.valid_index, self.nonvalid_index = self.get_valid_data(self.is_valid)
        # set weakly supervised parameters
        # ori_vallen = len(self.valid_index)
        # self.valid_index = self.valid_index[:int(ori_vallen/3)]
        # self.is_valid[self.valid_index[int(ori_vallen/3):]] = 0
        ############
        self.depth_maps1 = depth_files['data'][self.valid_index]
        self.segmentation1 = self.segmentation[self.valid_index]
        self.image_coordinates1 = labels['image_coordinates'][self.valid_index]
        self.real_world_coordinates1 = labels['real_world_coordinates'][self.valid_index]
        self.visible_joints1 = labels['visible_joints'][self.valid_index]
        self.id1 = labels['id'][self.valid_index]
        self.is_valid1 = labels['is_valid'][self.valid_index]
        
        self.valid_length = len(self.valid_index)
        self.id = labels['id']
        self.image_coordinates = labels['image_coordinates']
        self.real_world_coordinates = labels['real_world_coordinates']
        self.visible_joints = labels['visible_joints']


        # self.perm = np.arange(self.dataset_length)
        self.perm = np.arange(self.valid_length)
        self.perm0 = np.arange(self.dataset_length)
        if is_shuffle:
            np.random.shuffle(self.perm)
            np.random.shuffle(self.perm0)

        # # initilize the Z value regressor
        # Z_graph = tf.Graph()
        # with Z_graph.as_default():
        #     sess = tf.Session(graph=Z_graph)
        #     self.Predictor = Zscript.Zpred(sess, setting)

    def get_valid_data(self, is_valid):
        is_valid = is_valid[:].astype(np.int8)
        indx0 = np.squeeze(np.argwhere(is_valid == 1))
        indx1 = np.squeeze(np.argwhere(is_valid == 0))
        return indx0, indx1

    def checkConsis(self,index, setting, window_size=12):
        if window_size == 0:
            return True
        else:
            lef_ind = index-window_size
            rig_ind = index+window_size
            start = str(self.id[lef_ind],encoding="utf8")
            if start[:2].lstrip('0') == '':
                personID = 0
            else:
                personID = int(start[:2].lstrip('0'))
            if start[3:].strip('0') == '':
                frameID = 0
            else:
                frameID = int(start[3:].lstrip('0'))
            for i in range(lef_ind+1,rig_ind+1):
                tmp = str(self.id[i], encoding="utf8")
                if tmp[:2].lstrip('0') == '':
                    tmp_person = 0
                else:
                    tmp_person = int(tmp[:2].lstrip('0'))
                if tmp[3:].lstrip('0') == '':
                    tmp_frame = 0
                else:
                    tmp_frame = int(tmp[3:].lstrip('0'))
                if tmp_person != personID:
                    return False
                if tmp_frame != i-lef_ind+frameID:
                    return False
            return True

    def data_augumentation(self, dmap, joints_2d, joint_3d):
        Bsize = dmap.shape[0]
        img = np.zeros_like(dmap, dtype=np.float)
        new_joints_2d = np.zeros_like(joints_2d, dtype=np.float32)
        new_joints_3d = joint_3d
        randata = np.random.rand()
        increment = np.random.rand() * 3
        if randata < 0.5:
            for i in range(Bsize):
                depth = dmap[i]
                pcloud = utils.Image2Pcloud(depth)
                pcloud[:, 2] = pcloud[:, 2] + increment
                new_joints_3d[i, :, 2] = joint_3d[i, :, 2] + increment
                pose_3d = copy.copy(new_joints_3d[i])
                new_2d = utils.Pcloud2Image(pose_3d)
                new_2d = np.transpose(new_2d)
                new_joints_2d[i, ...] = new_2d[:, :2]
                p = utils.Pcloud2Image(pcloud)
                img[i, np.int32(p[1, :]), np.int32(p[0, :])] = p[2, :]

        else:
            img = dmap
            new_joints_3d = joint_3d
            new_joints_2d = joints_2d
        return img, new_joints_2d, new_joints_3d


    def next_batch(self, batch_size,setting, window_size=12):
        start = self.batch_offset
        self.batch_offset = self.batch_offset + batch_size*setting.window_size
        # self.batch_offset = self.batch_offset + batch_size
        out_dmap = np.zeros([batch_size,setting.window_size,setting.height, setting.width])
        out_imcor = np.zeros([batch_size,setting.window_size, setting.joint_num, 2])
        out_wocor = np.zeros([batch_size,setting.window_size, setting.joint_num, 3])
        out_valid = np.zeros([batch_size,setting.window_size])
        out_seg = np.zeros([batch_size, setting.window_size, setting.height, setting.width])

        id_tmp = np.zeros(shape=batch_size, dtype=np.int8)
        if self.batch_offset > self.dataset_length:
            # finish epoch
            self.epochs_completed += 1
            print("************ Epochs completed: " + str(self.epochs_completed) + "************")
            # shuffle the data
            np.random.shuffle(self.perm0)
            # start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        for i in range(batch_size):
            tmpidx = self.perm0[i * setting.window_size+start]
            # tmpidx = self.perm0[i + start]
            if tmpidx-window_size < 0:
                tmpidx = window_size
            if tmpidx+window_size > self.dataset_length-1:
                tmpidx = self.dataset_length-1-window_size
            if self.checkConsis(tmpidx, setting):
                HumanId = str(self.id[tmpidx], encoding="utf8")
                if HumanId[:2].lstrip('0') == '':
                    personID = 0
                else:
                    personID = int(HumanId[:2].lstrip('0'))
                id_tmp[i] = personID
                out_dmap[i] = self.depth_maps[tmpidx-window_size:tmpidx+window_size+1]
                out_imcor[i] = self.image_coordinates[tmpidx - window_size:tmpidx + window_size + 1]
                out_wocor[i] = self.real_world_coordinates[tmpidx - window_size:tmpidx + window_size + 1]
                out_valid[i] = self.is_valid[tmpidx - window_size:tmpidx + window_size + 1]
                out_seg[i] = self.segmentation[tmpidx - window_size:tmpidx + window_size + 1]

            else:
                return self.next_batch(batch_size,setting)
        return out_dmap, \
               out_imcor, \
               out_wocor, out_valid, id_tmp, out_seg

    def next_normalized_batch(self, batch_size, setting):
        dmap, image_coor, world_coor, out_valid, cur_id, seg = self.next_batch(batch_size, setting)


        seg = seg + 0.1
        seg = np.clip(seg, 0, 0.1) * 10
        

        if dmap.ndim == 4:
            time_step = dmap.shape[1]
            dmap = dmap.reshape([batch_size * time_step, dmap.shape[2], -1])
            image_coor = image_coor.reshape([batch_size * time_step, image_coor.shape[2], -1])
            world_coor = world_coor.reshape([batch_size * time_step, world_coor.shape[2], -1])
            out_valid = out_valid.reshape([batch_size * time_step])

       
        coor_x = np.expand_dims(np.clip(image_coor[:, :, 0], 0, 319),
                       axis=2)
        coor_y = np.expand_dims(np.clip(image_coor[:, :, 1], 0, 239),
                                axis=2)
        image_coor = np.concatenate([coor_x, coor_y], axis=2)
        seg = np.reshape(seg, [-1, setting.height, setting.width])


        image_coor = np.floor(image_coor)
        return dmap, image_coor, world_coor, out_valid, cur_id, seg


    
    def next_demo_batch(self,batch_size, setting):
        start = self.batch_offset
        self.batch_offset = self.batch_offset + batch_size*setting.window_size
        if self.batch_offset > self.dataset_length:
            # finish epoch
            self.epochs_completed += 1
            print("************ Epochs completed: " + str(self.epochs_completed) + "************")
            start = 0
            self.batch_offset = batch_size*setting.window_size
        end = self.batch_offset
        tmpidx = np.arange(start,end)
        dmap = self.depth_maps[tmpidx]
        dmap = dmap.reshape([batch_size, setting.window_size,setting.height,setting.width])
        color_maps = self.color_maps[tmpidx].reshape([batch_size, setting.window_size,setting.height,setting.width, 3])
        return dmap, color_maps, self.depth_2d[tmpidx].reshape([batch_size, setting.window_size, 15, 2])

    def next_demo_norm_batch(self, batch_size, setting):
        dmap, color_maps, depth_2d = self.next_demo_batch(batch_size, setting)
        flag = False
        if dmap.ndim == 4:
            flag = True
            time_step = dmap.shape[1]
            dmap = dmap.reshape([batch_size * time_step, dmap.shape[2], -1])
            color_maps = color_maps.reshape([batch_size * time_step, setting.height, setting.width, 3])
            depth_2d =depth_2d.reshape([batch_size*time_step, setting.joint_num, 2])
            


        tmp_a = np.expand_dims(np.clip(depth_2d[:, :, 0], 0 + setting.sample_pixel, 319 - setting.sample_pixel),
                               axis=2)
        tmp_b = np.expand_dims(np.clip(depth_2d[:, :, 1], 0 + setting.sample_pixel, 239 - setting.sample_pixel),
                               axis=2)

        bbox_set = utils.GetBatch2DBBoxSet(np.concatenate([tmp_a, tmp_b], axis=2), setting)
        pcloud_np = utils.GetEveryPcloud(dmap, bbox_set, setting, is_demo='demo')



        pcloud_z = pcloud_np[:, 8, ...]
        mean_z = np.zeros(pcloud_z.shape[0], dtype=np.float32)
        for i in range(pcloud_z.shape[0]):
            mean_z[i] = np.mean(pcloud_z[i, pcloud_z[i, :, 2] > 0, 2])
       
        pose_3d_init_np = utils.Imgcoor2Pcloud(dmap, np.concatenate([tmp_b, tmp_a], axis=2), is_demo='demo', mean_z = mean_z)

        # Get the point cloud bounding box
        pose_3d_init_np_a = pose_3d_init_np

        # center = pcloud_np.reshape([batch_size * setting.window_size, -1, 3])
        # center = np.mean(center, axis=1)
        center = pose_3d_init_np_a[:, 8, :]

        bbox = np.concatenate([center, np.tile(setting.bbox_size, [batch_size * setting.window_size, 1])], axis=1)
        bbox = np.tile(np.expand_dims(bbox, axis=1), [1, setting.joint_num, 1])

        # Point cloud normalization
        pcloud_np_n, a = utils.PcloudNomalization(pcloud_np, bbox)
        pose_3d_init_np_a = np.expand_dims(pose_3d_init_np_a, axis=2)
        pose_3d_init_np_n, a = utils.PcloudNomalization(pose_3d_init_np_a, bbox)
        pose_3d_init_np_n = np.squeeze(pose_3d_init_np_n, axis=2)

        return pcloud_np_n, pose_3d_init_np_n, bbox, depth_2d


    def next_output_batch(self,batch_size):
        start = self.batch_offset
        self.batch_offset = self.batch_offset + batch_size
        if self.batch_offset > self.dataset_length:
            # finish epoch
            self.epochs_completed += 1
            print("************ Epochs completed: " + str(self.epochs_completed) + "************")
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        tmpidx = np.arange(start,end)
        return self.depth_maps1[tmpidx], self.real_world_coordinates1[tmpidx]


    def next_selected_batch(self, index, setting):
        dmap = []
        image_coor = []
        world_coor = []
        out_valid = []
        seg = []

        for i in range(len(index)):
            dmap.append(self.depth_maps[index[i]])
            image_coor.append(self.image_coordinates[index[i]])
            world_coor.append(self.real_world_coordinates[index[i]])
            out_valid.append(self.is_valid[index[i]])
            seg.append(self.segmentation[index[i]])
        dmap = np.array(dmap)
        image_coor = np.array(image_coor)
        world_coor = np.array(world_coor)
        out_valid = np.array(out_valid)
        seg = np.array(seg)

        # segmentation weight map
        seg = seg + 0.1
        seg = np.clip(seg, 0, 0.1) * 10

        if dmap.ndim == 4:
            time_step = dmap.shape[1]
            dmap = dmap.reshape([batch_size * time_step, dmap.shape[2], -1])
            image_coor = image_coor.reshape([batch_size * time_step, image_coor.shape[2], -1])
            world_coor = world_coor.reshape([batch_size * time_step, world_coor.shape[2], -1])
            out_valid = out_valid.reshape([batch_size * time_step])


        # joint_coor = np.array(joint_coor, dtype=np.int32)
        coor_x = np.expand_dims(np.clip(image_coor[:, :, 0], 0, 319),
                       axis=2)
        coor_y = np.expand_dims(np.clip(image_coor[:, :, 1], 0, 239),
                                axis=2)
        image_coor = np.concatenate([coor_x, coor_y], axis=2)
        seg = np.reshape(seg, [-1, setting.height, setting.width])


        image_coor = np.floor(image_coor)
        return dmap, image_coor, world_coor, out_valid, seg




if __name__ == "__main__":
    
    joint_2d = np.ones([23,15])
    mask = creat_joint2d_noise(joint_2d)
    print(mask)