import numpy as np

batch_size = 1
learning_rate_base = 5e-4
decay_steps = 1000
decay_rate = 0.95
learning_rate_min = 0.00001
weight = 1e-5

# basic pcloud parameters
height = 424
width = 512
joint_num = 25
window_size = 61
sample_pixel = 8
simplified_num = 56

initial_temperature = 0.01
zsample_pixel = 20
zsamples = (2 * zsample_pixel) * (2 * zsample_pixel)
sample_pixel_total = (2 * sample_pixel) * (2 * sample_pixel)
bbox_size = np.array([1.8, 2, 1.5])

#
rot_temp = np.array([[[1 / 0.0027, 0], [0, -1 / 0.0027], [260.4444, 210.1111]]], dtype=np.float32)
rotmat = np.tile(rot_temp, (batch_size*window_size, 1, 1))

demo_rot = np.array([[[121.1511, 0], [0, -109.6735], [145.8649, 114.5711]]], dtype=np.float32)
demomat = np.tile(demo_rot, (batch_size*window_size, 1, 1))

# skeleton parameter

skeleton_id = [[1,2], [2,4], [4,6],
               [1,3], [3,5], [5,7],
               [8,9], [9,11], [11,13],
               [8,10], [10,12], [12,14], [0, 1], [1, 8]]

bonearray = np.array([[0,1],[1,20],[20,8],[20,4],[8,9],[9,10],[10,11],[11,24],[11,23],[20,2],
             [2,3],[4,5],[5,6],[6,7],[7,22],[7,21],[0,16],[16,17],[17,18],[18,19],[0,12],
                      [12,13],[13,14],[14,15]])

# NTU bonearray
# bonearray1 = np.zeros([25,24])
# for i in range(24):
#     bonearray1[bonearray[i,0],i] = -1
#     bonearray1[bonearray[i,1],i] = 1

# ITOP bonearray
bonearray1 = np.zeros([15,14])
for i in range(14):
    bonearray1[skeleton_id[i][0],i] = -1
    bonearray1[skeleton_id[i][1],i] = 1

skeleton = [0.18099216, 0.30136532, 0.3083691, 0.1809909, 0.30517647, 0.3111792, 0.26156965,
            0.4784025, 0.46032718, 0.26156482, 0.47747827, 0.4607155, 0.23529877, 0.23755938]

ratio_id = [[0, 1], [1, 2],
            [3, 4], [4, 5],
            [6, 7], [7, 8],
            [9, 10], [10, 11],
            [0, 3], [1, 4],
            [2, 5], [6, 9],
            [7, 10], [8, 11], [12, 13], [12, 0], [12, 3]]

ratio = [0.5888, 0.9935, 0.5870, 0.9930, 0.5475, 1.0404, 0.5482, 1.0384, 1, 1, 1, 1, 1, 0.9994, 0.9816, 1.3014, 1.3014]


