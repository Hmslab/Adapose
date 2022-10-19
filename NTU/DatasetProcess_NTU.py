import numpy as np
import cv2

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True
    return missing_files

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])

    cursor = 0
    body_array = []
    for frame in range(nframe):
        bodymat = dict()
        bodymat['file_name'] = file_path[-29:-9]
        nbody = int(datas[1][:-1])
        bodymat['nbodys'] = []
        bodymat['njoints'] = njoints
        bodymat['cframe'] = frame

        for body in range(max_body):
            if save_skelxyz:
                bodymat['skel_body{}'.format(body)] = np.zeros(shape=(njoints, 3))
            if save_rgbxy:
                bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(njoints, 2))
            if save_depthxy:
                bodymat['depth_body{}'.format(body)] = np.zeros(shape=(njoints, 2))
        # above prepare the data holder

        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
            # skip the empty frame
        bodymat['nbodys'] = bodycount
        # bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][joint] = jointinfo[5:7]
        # prune the abundant bodys
        for each in range(max_body):
            if each >= bodymat['nbodys']:
                if save_skelxyz:
                    del bodymat['skel_body{}'.format(each)]
                if save_rgbxy:
                    del bodymat['rgb_body{}'.format(each)]
                if save_depthxy:
                    del bodymat['depth_body{}'.format(each)]
        body_array.append(bodymat)
    return body_array

def _read_skeleton_info(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])

    cursor = 0
    body_array = []
    file_name_array = []
    for frame in range(nframe):
        # name for the filename
        # c for the current frame
        # n for the number of body in the frame
        bodymat = dict()
        bodymat['name'] = file_path[-29:-9]
        nbody = int(datas[1][:-1])
        bodymat['f'] = frame
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        bodymat['n'] = bodycount
        if bodycount == 0:
            continue
            # skip the empty frame
        for body in range(bodycount):
            cursor += 1
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
        body_array.append(bodymat)
    return body_array

def _read_skeleton_selected(file_path, indx, window_size=5):
    # return:
    #       - array of 3D joint position
    #       - array of 2D joint position in depth image
    body = _read_skeleton(file_path)
    frame_count = len(body)
    jpos_3d = np.zeros(shape=[2 * window_size + 1, body[0]['njoints'], 3], dtype=np.float)
    jpos_2d = np.zeros(shape=[2 * window_size + 1, body[0]['njoints'], 2], dtype=np.int)
    if indx-window_size < 0:
        indx = window_size
    if indx + window_size > frame_count-1:
        indx = frame_count - window_size - 1
    cont = 0
    for i in range(-window_size, window_size+1):
        c_body = body[indx + i]
        jpos_2d[cont, ...] = c_body['depth_body0']
        jpos_3d[cont, ...] = c_body['skel_body0']
        cont += 1
    return jpos_2d, jpos_3d, frame_count


def _read_dmap_selected(file_path, indx, frame_num, window_size=5):
    dmap_array = np.zeros(shape=[window_size*2+1, 424, 512],dtype=np.float)

    if indx - window_size < 1:
        indx = window_size + 1
    if indx + window_size > frame_num:
        indx = frame_num - window_size

    cont = 0
    for i in range(-window_size, window_size+1):
        image_path = file_path + '/MDepth-%08d.png' % (indx + i)
        c_dmap = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH).astype(np.float)
        dmap_array[cont, ...] = c_dmap / 1000
        cont += 1
    return dmap_array



