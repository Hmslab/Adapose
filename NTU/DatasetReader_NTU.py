import os
import numpy as np
from datetime import datetime
import DatasetProcess as dp
import utils.np_utils as utils
import sys
sys.setrecursionlimit(90000)


class DatasetReader:
    dataset_length = 0
    batch_offset = 0
    epochs_completed = 0
    height = 424
    width = 512
    body_index = 6 #window_size
    perm = None
    body = None

    def __init__(self, dataset_path, mode, is_shuffle=True, is_mask=True):
        self.data_path = dataset_path
        self.is_mask = True
        # skeleton data
        # for debug use: skel_info_test.txt, otherwise use: skel_info.txt
        if os.path.exists(dataset_path + 'sample_'+mode+'_info.txt'):
            file = open(dataset_path + 'sample_'+mode+'_info.txt', 'r')
            self.body = eval(file.read())
            file.close()
        else:
            skel_path = dataset_path + 'nturgb+d_skeletons/'
            self.body = []
            count = 0
            file = open(dataset_path + 'skel_info.txt', 'w')
            tic = datetime.now()
            for filename in os.listdir(skel_path):
                if count > 10:
                    break
                temp_body = dp._read_skeleton_info(skel_path + filename)
                self.body.extend(temp_body)
                count += 1
                print('Processing ' + filename)
            toc = datetime.now()
            file.write(str(self.body))
            print('Elapsed time: %f seconds' % (toc - tic).total_seconds())
        self.dataset_length = len(self.body)
        self.perm = np.arange(self.dataset_length)
        if is_shuffle:
            np.random.shuffle(self.perm)
        self.tmpidx = np.squeeze(self.perm[np.random.randint(0, self.dataset_length, 1)])


    def next_batch(self, batch_size, setting):
        window_size = int((setting.window_size - 1) / 2)
        # get the index
        start = self.batch_offset
        self.batch_offset = self.batch_offset + batch_size
        if self.batch_offset > self.dataset_length:
            # finish epoch
            self.epochs_completed += 1
            print("************ Epochs completed: " + str(self.epochs_completed) + "************")
            # shuffle the data
            np.random.shuffle(self.perm)
            # start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        tmpidx = self.perm[start:end]

        action_int = np.zeros([batch_size], dtype=np.int8)
        # prepare the data array
        dmap_array = np.zeros(shape=[batch_size, 2 * window_size + 1, 424, 512])
        jpos_2d_array = np.zeros(shape=[batch_size, 2 * window_size + 1, setting.joint_num, 2])
        jpos_3d_array = np.zeros(shape=[batch_size, 2 * window_size + 1, setting.joint_num, 3])

        # load the data from file
        for i in range(batch_size):
            c_body_indx = self.body[tmpidx[i]]
            action = c_body_indx['name'].find('A')
            # if int(c_body_indx['name'][action+1:action+4].strip('0')) not in [99,67,9,8,24,23,63]:
            if int(c_body_indx['name'][action + 1:action + 4].strip('0')) not in [40, 80, 9, 8, 24, 96, 97, 98, 102]:
                return self.next_batch(batch_size, setting)
            else:
                action_int[i] = int(c_body_indx['name'][action + 1:action + 4].strip('0'))
                filename_d = self.data_path + 'nturgb+d_depth_masked/' + c_body_indx['name']
                filename_s = self.data_path + 'nturgb+d_skeletons/' + c_body_indx['name'] + '.skeleton'
                if not os.path.exists(filename_d + '/'):
                    return self.next_batch(batch_size, setting)
                else:
                    c_body_2d, c_body_3d, frame_num = dp._read_skeleton_selected(filename_s, c_body_indx['f'],
                                                                                 window_size)
                    if frame_num < setting.window_size:
                        return self.next_batch(batch_size, setting)
                    else:
                        dmap_array[i, ...] = dp._read_dmap_selected(filename_d, c_body_indx['f'] + 1, frame_num,
                                                                    window_size)
                        jpos_2d_array[i, ...] = c_body_2d
                        jpos_3d_array[i, ...] = c_body_3d

        return dmap_array, jpos_2d_array, jpos_3d_array, action_int

    def next_normalized_batch(self, batch_size, setting):
        dmap, image_coor, world_coor, action_int = self.next_batch(batch_size, setting)
        if dmap.ndim == 4:
            time_step = dmap.shape[1]
            dmap = dmap.reshape([batch_size*time_step, dmap.shape[2],-1])
            image_coor = image_coor.reshape([batch_size*time_step, image_coor.shape[2],-1])
            world_coor = world_coor.reshape([batch_size*time_step, world_coor.shape[2], -1])

        return dmap, image_coor, world_coor, action_int