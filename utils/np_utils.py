import numpy as np
# import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy import io


def depth_map_to_image(depth_map, joints=None, joints_true=None, type=cv2.COLORMAP_OCEAN):
    img = (depth_map - depth_map.min())/(depth_map.max()-depth_map.min())
    img = np.array(img * 255, dtype=np.uint8)
    # HOT AUTUMN HSV
    img = cv2.applyColorMap(img, type)
    if joints is not None:
        for j in range(15):
            x, y = joints[j, 0], joints[j, 1]
            cv2.circle(img, (x, y), 1, (0, 0, 0), thickness=2)
            x, y = joints_true[j, 0], joints_true[j, 1]
            cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=2)
    return img


def Dmap2img(dmap):
    glimpse = cv2.normalize(dmap, dmap, 0, 1, cv2.NORM_MINMAX)
    glimpse = np.array(glimpse * 255, dtype=np.uint8)
    glimpse = cv2.cvtColor(glimpse, cv2.COLOR_GRAY2RGB)
    glimpse = cv2.applyColorMap(glimpse, cv2.COLORMAP_OCEAN)
    return glimpse

def Image2Pcloud(img):
    shape = img.shape
    c = 0.0035
    fir = np.tile(np.expand_dims(np.arange(shape[1]), 1), [1, shape[0]]).reshape([1, -1])
    sec = np.arange(shape[0])
    sec = np.tile(sec, [shape[1]])
    fir = np.squeeze(fir)
    z = img[sec, fir]
    temp = [(fir-160)*c * z, (120 - sec)*c * z, z]
    temp = np.transpose(np.array(temp, dtype=np.float))
    return temp

def Pcloud2Image(pcloud):
    pcloud = np.transpose(pcloud)
    Matrix = np.array([[285.7143,0,160.0000],[0,-285.7143,120.0000],[0,0,1]])
    zp = np.matmul(Matrix,pcloud)
    p = zp
    p[:2, zp[2,:] != 0] = np.multiply(zp[:2,zp[2,:] != 0], 1/zp[2,zp[2,:] != 0])
    return p

def ConvertDmap(img):
    shape = img.shape
    pcloud = np.ones([shape[0] * shape[1], 3], dtype=np.float)
    c = 0.0035
    indx = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            z = img[i, j]
            print('i=%3d j=%3d z=%3f x=%f y=%f z=%f' % (i, j, z, (j-160) * c * z, (120 - i) * c * z, z))
            temp = [(j-160) * c * z, (120 - i) * c * z, z]
            temp = np.array(temp, dtype=np.float)
            pcloud[indx, :] = temp
            indx = indx + 1
    print('=================================')
    return pcloud


def ConvertDmap_BBox(img, xmin, xmax, ymin, ymax):
    shape = img.shape
    pcloud = np.zeros([shape[0] * shape[1], 3], dtype=np.float)
    c = 0.0035
    for i in range(int(xmin)-20, int(xmax)+20):
        for j in range(int(ymin)-20, int(ymax)+12):
            indx = i * shape[0] + j
            z = img[i, j]
            temp = [(j-160) * c * z, (120 - i) * c * z, z]
            temp = np.array(temp, dtype=np.float)
            pcloud[indx, :] = temp
    return pcloud


def Imgcoor2Pcloud(img, coor, is_demo='ITOP', mean_z=None):
    c = 0.0035
    pcloud = np.zeros([coor.shape[0], coor.shape[1], 3])
    for i in range(0, coor.shape[0]):
        temp_img = img[i, :, :]
        m = coor[i, :, 0].astype(int)
        n = coor[i, :, 1].astype(int)
        if is_demo == 'demo':
            if z == 0: z = 2.3
            temp = [(0.0083*n - 1.2040) * z, (-0.0091*m + 1.0447) * z, np.tile(z,len(m))]
        elif is_demo == 'ITOP':
            if mean_z is not None:
                z = mean_z[i]
            else:
                z = temp_img[m, n]
            temp = [(n - 160) * c * z, (120 - m) * c * z, z]
        elif is_demo == 'NTU':
            temp = [(0.0027 * n -0.7032) * z, (-0.0027 * m + 0.5673) * z, np.tile(z,len(m))]
        else:
            temp = [(0.0039 * n - 0.6198) * z, (0.0039 * m - 0.4648) * z, np.tile(z,len(m))]
        temp = np.array(temp, dtype=np.float)
        pcloud[i, :, :] = np.transpose(temp)
    return pcloud


def ConvertDmap_Patch(img, offset):
    shape = img.shape
    pcloud = np.zeros([shape[0] * shape[1], 3], dtype=np.float)
    c = 0.0035
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            indx = i * shape[0] + j
            z = img[i, j]
            temp = [(offset[1] + j - 160) * c * z, (120 - i - offset[0]) * c * z, z]
            temp = np.array(temp, dtype=np.float)
            pcloud[indx, :] = temp
    return pcloud


def ConvertDmap_Patch_BBox(img, xmin, xmax, ymin, ymax, is_demo='ITOP'):
    shape = img.shape
    pcloud = np.zeros([int(xmax - xmin) * int(ymax - ymin), 3], dtype=np.float)
    c = 0.0035
    indx = 0
    if is_demo == 'demo':
        for i in range(int(xmin), int(xmax)):
            for j in range(int(ymin), int(ymax)):
                z = img[j, i]
                temp = [(0.0083 * i - 1.2040) * z, (-0.0091 * j + 1.0447) * z, z]
                temp = np.array(temp, dtype=np.float)
                pcloud[indx, :] = temp
                indx = indx + 1
    elif is_demo == 'ITOP':
        for i in range(int(xmin), int(xmax)):
            for j in range(int(ymin), int(ymax)):
                z = img[j, i]
                temp = [(i - 160) * c * z, (120 - j) * c * z, z]
                temp = np.array(temp, dtype=np.float)
                pcloud[indx, :] = temp
                indx = indx + 1
    elif is_demo == 'NTU':
        fir = np.tile(np.expand_dims(np.arange(int(xmin), int(xmax)),1),[1,int(ymax)-int(ymin)]).reshape([1,-1])
        sec = np.arange(int(ymin), int(ymax))
        sec = np.tile(sec,[int(xmax)-int(xmin)])
        fir = np.squeeze(fir)
        z = img[sec, fir]
        temp = [(0.0027 * fir - 0.7032) * z, (-0.0027 * sec + 0.5673) * z, z]
        temp = np.transpose(np.array(temp, dtype=np.float))
        pcloud = temp

    else:
        for i in range(int(xmin), int(xmax)):
            for j in range(int(ymin), int(ymax)):
                z = img[j, i]
                temp = [(0.0039 * i - 0.6198) * z, (0.0039 * j - 0.4648) * z, z]
                temp = np.array(temp, dtype=np.float)
                pcloud[indx, :] = temp
                indx = indx + 1
    return pcloud

def ConvertDmap_Patch_BBox_new(img, bbox_set, is_demo='ITOP'):
    c = 0.0035
    xlen = int(bbox_set[0,0,1] - bbox_set[0,0,0])
    ylen = int(bbox_set[0, 0, 3] - bbox_set[0, 0, 2])
    xmin = bbox_set[:, :, 0].astype(int)
    xmax = bbox_set[:, :, 1].astype(int)
    ymin = bbox_set[:, :, 2].astype(int)
    ymax = bbox_set[:, :, 3].astype(int)
    fir = np.zeros([xmin.shape[0],xmin.shape[1],ylen*xlen],dtype=np.int)
    sec = np.zeros([ymin.shape[0],ymin.shape[1],xlen*ylen],dtype=np.int)
    z = np.zeros([xmin.shape[0],xmin.shape[1],ylen*xlen],dtype=np.float)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            fir[i,j,:] = np.squeeze(np.tile(np.expand_dims(np.arange(xmin[i,j],xmax[i,j]),1),[1,ylen]).reshape([1,-1]))
            sec[i,j,:] = np.tile(np.arange(ymin[i,j], ymax[i,j]),[xlen])
            z[i,j,:] = img[i,sec[i,j,:],fir[i,j,:]]

    if is_demo == 'NTU':
        temp = [(0.0027 * fir - 0.7032) * z, (-0.0027 * sec + 0.5673) * z, z]  #3*B*J*sample_num
        temp = np.transpose(np.array(temp, dtype=np.float32), [1, 2, 3, 0])
    if is_demo == 'ITOP':
        temp = [(fir - 160) * c * z, (120 - sec) * c * z, z]
        temp = np.transpose(np.array(temp, dtype=np.float32), [1, 2, 3, 0])
    return temp

def PCloud2Voxel(pcloud, grid_size):
    shape = pcloud.shape

    xmin = np.min(pcloud[:, 0])
    ymin = np.min(pcloud[:, 1])
    zmin = np.min(pcloud[:, 2])
    zmin = 1.5
    xmax = np.max(pcloud[:, 0])
    ymax = np.max(pcloud[:, 1])
    zmax = np.max(pcloud[:, 2])
    x_step = (xmax - xmin) / (grid_size[0] - 1)
    y_step = (ymax - ymin) / (grid_size[1] - 1)
    z_step = (zmax - zmin) / (grid_size[2] - 1)

    voxel_point = np.zeros([grid_size[0], grid_size[1], grid_size[2]], dtype=np.float)
    for i in range(0, shape[0]):
        point = pcloud[i, :]
        if point[2] < zmin:
            continue
        else:
            x_index = int(np.fix((point[0] - xmin) / x_step))
            y_index = int(np.fix((point[1] - ymin) / y_step))
            z_index = int(np.fix((point[2] - zmin) / z_step))
            voxel_point[x_index, y_index, z_index] = 1

    return voxel_point


def glimpseSensor(img, labels):
    depth = 4
    minRadius = 10
    maxRadius = minRadius * (2 ** (depth - 1))
    offset = maxRadius

    image_list = []
    image_list.append(img)
    for i in range(1, depth):
        k_size = 2 ** i
        kernel = get_GuassianKernal(k_size, 2 * k_size)
        img_temp = cv2.filter2D(img, -1, kernel)
        image_list.append(img_temp)

    glimpse = image_list[depth - 1]
    for j in range(0, depth):
        i = depth - j - 1
        img_temp = image_list[i]
        radius = int(minRadius * (2 ** i))
        adjusted_loc = labels - radius
        glimpse[adjusted_loc[0]: adjusted_loc[0] + radius * 2, adjusted_loc[1]: adjusted_loc[1] + radius * 2] = img_temp[adjusted_loc[0]: adjusted_loc[0] + radius * 2, adjusted_loc[1]: adjusted_loc[1] + radius * 2]
    loc = labels - maxRadius
    loc[0] = np.clip(loc[0], 0, img.shape[0] - maxRadius * 2)
    loc[1] = np.clip(loc[1], 0, img.shape[1] - maxRadius * 2)
    max_glimpse = glimpse[loc[0]: loc[0] + maxRadius * 2, loc[1]: loc[1] + maxRadius * 2]

    loc = labels - minRadius
    loc[0] = np.clip(loc[0], 0, img.shape[0] - minRadius * 2)
    loc[1] = np.clip(loc[1], 0, img.shape[1] - minRadius * 2)
    min_glimpse = glimpse[loc[0]: loc[0] + minRadius * 2, loc[1]: loc[1] + minRadius * 2]
    return max_glimpse, min_glimpse, loc


def img_process(imgs, labels, voxel_szie):
    glimpse = np.zeros([imgs.shape[0], 15, 160, 160], dtype=np.float32)
    voxel = np.zeros([imgs.shape[0], 15, voxel_szie, voxel_szie, voxel_szie], dtype=np.int)
    for i in range(0, imgs.shape[0]):
        for j in range(0, labels.shape[1]):
            max_glimpse, min_glimpse, loc = glimpseSensor(imgs[i, :, :], labels[i, j, :])
            pcloud = ConvertDmap_Patch(min_glimpse, loc)
            temp_voxel = PCloud2Voxel(pcloud, [voxel_szie, voxel_szie, voxel_szie])
            glimpse[i, j, :, :] = max_glimpse
            voxel[i, j, :, :, :] = temp_voxel
    return glimpse, voxel


def get_GuassianKernal(size, sigma, is_normal=True):
    arr = np.zeros((2*size-1, 2*size-1))
    sum = 0
    for i in range(0, 2*size-1):
        for j in range(0, 2*size-1):
            arr[i, j] = np.exp(-((i-size)*(i-size)+(j-size)*(j-size)) / (2*sigma*sigma))
    if is_normal:
        norm = np.sum(arr, dtype=np.float)
        arr = arr / norm
    return arr


def PointProject(point):
    '''

    :param point: input point cloud, shape:[batch_size, N, 3]
    :return: img_point: output image coordinate, shape:[batch_size, N, 2]
    '''
    C = 0.0035
    intri_mat = np.array([[1 / C, 0], [0, -1 / C], [160, 120]])
    img_coor = np.zeros([point.shape[0], point.shape[1], 3], dtype=np.float32)
    for i in range(0, point.shape[0]):
        for j in range(0, point.shape[1]):
            z = point[i, j, 2]
            if z == 0:
                continue
            else:
                cur_point = point[i, j, :] / z
                img_coor[i, j, 0:2] = np.matmul(cur_point, intri_mat)
                img_coor[i, j, 2] = z
    img_point = np.round(img_coor).astype(np.int)

    return img_point


def Imgcoor2Img(img_coor):
    img = np.zeros([240, 320])
    for i in range(0, 240):
        for j in range(0, 320):
            indx = i * 240 + j
            if (img_coor[indx, 0:1] == [0, 0]).all():
                continue
            else:
                img[i, j] = img_coor[indx, 2]
    return  img


def DetctionProcess(image, annotation):
    image = np.expand_dims(image, axis=3)
    new_annotations = np.zeros([image.shape[0], image.shape[1], image.shape[2], annotation.shape[1]])
    for i in range(0, image.shape[0]):
        for j in range(0, annotation.shape[1]):
            indx = annotation[i, j, :]
            indx[1] = np.clip(indx[1], 0, image.shape[1]-1)
            indx[0] = np.clip(indx[0], 0, image.shape[2]-1)
            new_annotations[i, indx[1], indx[0], j] = 1
            kernel = get_GuassianKernal(10, 2, False)
            temp = cv2.filter2D(new_annotations[i, :, :, j], -1, kernel)
            new_annotations[i, :, :, j] = temp
    return image, new_annotations


def PoseDetectionVisulization(image, annotation_gt, annotation_pred):
    img = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    xmin, xmax, ymin ,ymax = 1000 , 0, 1000, 0
    idx_xmin, idx_xmax, idx_ymin, idx_ymax = 1000, 0, 1000, 0
    count = 0
    threshold = 30
    cv2.imwrite('ts_image.png', img)

    for j in range(annotation_pred.shape[2]):
        x, y = annotation_gt[j, 0], annotation_gt[j, 1]
        if x < xmin : xmin = x
        if x > xmax : xmax = x
        if y < ymin : ymin = y
        if y > ymax : ymax = y
        cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=2)

        heat_map = annotation_pred[:, :, j]
        indx = np.where(heat_map==np.max(heat_map))

        if indx[1] < idx_xmin : idx_xmin = indx[1]
        if indx[1] > idx_xmax : idx_xmax = indx[1]
        if indx[0] < idx_ymin : idx_ymin = indx[0]
        if indx[0] > idx_ymax : idx_ymax = indx[0]

        cv2.circle(img, (indx[1], indx[0]), 1, (0, 0, 0), thickness=2)
    if idx_ymax + 12 > 240:
        idx_ymax = 227
    cv2.rectangle(img, (idx_xmin - 20, idx_ymin - 20), (idx_xmax + 20, idx_ymax + 12), (0, 0, 255))

    return img, count, xmin, xmax, ymin, ymax

def Pcloud2Image_new(pcloud):
    pcloud = np.transpose(pcloud)
    Matrix = np.array([[285.7143,0,160.0000],[0,-285.7143,120.0000],[0,0,1]])
    zp = np.matmul(Matrix,pcloud[:3, :])
    p = zp
    p[:2, zp[2,:] != 0] = np.multiply(zp[:2,zp[2,:] != 0], 1/zp[2,zp[2,:] != 0])
    result = np.zeros_like(pcloud)
    result[:3, :] = p
    result[3, :] = pcloud[3, :]
    return result

def generate_joint_mask(segmentation, joint_annotation):
    joint_mask = copy.copy(segmentation)
    kernel = get_GuassianKernal(10, 2, False)
    for i in range(0, joint_annotation.shape[0]):
        for j in range(0, joint_annotation.shape[1]):
            for k in range(0, joint_annotation.shape[2]):
                indx = joint_annotation[i, j, k, :].astype(np.int)
                indx[1] = np.clip(indx[1], 0, segmentation.shape[2] - 1)
                indx[0] = np.clip(indx[0], 0, segmentation.shape[3] - 1)
                joint_mask[i, j, indx[1], indx[0]] = 1
                temp = cv2.filter2D(joint_mask[i, j, :, :], -1, kernel)
                joint_mask[i, j, :, :] = temp
    return joint_mask

def PoseVisualization(image, annotation_gt, annotation_pred):
    img = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    xmin, xmax, ymin ,ymax = 1000 , 0, 1000, 0
    idx_xmin, idx_xmax, idx_ymin, idx_ymax = 1000, 0, 1000, 0

    for j in range(annotation_pred.shape[1]):
        x, y = annotation_gt[0, j, 0], annotation_gt[0, j, 1]


    for j in range(annotation_pred.shape[1]):
        a, b = annotation_pred[0, j, 0], annotation_pred[0, j, 1]


    return img


def GetCoarsePose(annotation_pred):
    result = np.ones([annotation_pred.shape[0], annotation_pred.shape[3], 3])
    for i in range(annotation_pred.shape[0]):
        for j in range(annotation_pred.shape[3]):
            heat_map = annotation_pred[i, :, :, j]
            indx = np.where(heat_map == np.max(heat_map))
            indx = np.reshape(indx, [2, -1])
            indx = np.transpose(indx)
            indx = indx[0, :]
            result[i, j, 0] = int(indx[0])
            result[i, j, 1] = int(indx[1])
    return result.astype(np.int)


def GetBatch2DBBox(pose_2d):
    bbox = np.zeros((pose_2d.shape[0], 4))
    for i in range(pose_2d.shape[0]):
        bbox[i, 0] = np.min(pose_2d[i, :, 0]) - 20
        bbox[i, 1] = np.max(pose_2d[i, :, 0]) + 20
        bbox[i, 2] = np.min(pose_2d[i, :, 1]) - 20
        bbox[i, 3] = np.max(pose_2d[i, :, 1]) + 12
        if bbox[i, 3] >= 240:
            bbox[i, 3] = 239
        if bbox[i, 1] >= 320:
            bbox[i, 1] = 319

    return bbox

def _pred2coor(pred_numpy,setting):
    pred_coor = np.zeros([pred_numpy.shape[0], setting.joint_num, 2],dtype=np.float32)
    for i in range(pred_numpy.shape[0]):  # batch_size
        for j in range(pred_numpy.shape[3]): #joint_num
            heat_map = pred_numpy[i, :, :, j]
            index = np.where(heat_map == np.max(heat_map))
            pred_coor[i, j, 0] = index[1]
            pred_coor[i, j ,1] = index[0]
    return pred_coor



def GetBatch2DBBoxSet(pose_2d,setting):
    bbox_set = np.zeros([pose_2d.shape[0],setting.joint_num,4])
    for i in range(pose_2d.shape[0]):
        for j in range(pose_2d.shape[1]):
            bbox_set[i,j,0] = pose_2d[i, j, 0]-setting.sample_pixel
            bbox_set[i,j,1] = pose_2d[i, j, 0]+setting.sample_pixel
            bbox_set[i,j,2] = pose_2d[i, j, 1]-setting.sample_pixel
            bbox_set[i,j,3] = pose_2d[i, j, 1]+setting.sample_pixel

    return bbox_set

def Getcorebox(joint_root,setting):
    bbox_set = np.zeros([joint_root.shape[0],4])
    for i in range(joint_root.shape[0]):
        bbox_set[i,0] = joint_root[i, 0]-setting.zsample_pixel
        bbox_set[i,1] = joint_root[i, 0]+setting.zsample_pixel
        bbox_set[i,2] = joint_root[i, 1]-setting.zsample_pixel
        bbox_set[i,3] = joint_root[i, 1]+setting.zsample_pixel
    return bbox_set




def GetSampledPcloud(depth_maps, bbox, setting):
    result = np.zeros((depth_maps.shape[0], setting.sample_num, 3))
    for itr in range(depth_maps.shape[0]):
        tmp_depth = depth_maps[itr, ...]
        xmin = bbox[itr, 0]
        xmax = bbox[itr, 1]
        ymin = bbox[itr, 2]
        ymax = bbox[itr, 3]
        pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax)

        # TODO: implement the KMeans sampling methods
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        points = [[i, j, k] for i, j, k in zip(x, y, z)]
        y_pred = KMeans(n_clusters=2).fit_predict(points)
        length = int(len(y_pred))
        count_one = 0
        for i in range(length):
            if y_pred[i] == 1: count_one = count_one + 1
        count_zero = length - count_one
        pcloud_one = np.zeros([count_one, 3], dtype=np.float)
        pcloud_zero = np.zeros([count_zero, 3], dtype=np.float)
        j, k = 0, 0
        for i in range(length):
            if int(y_pred[i]) == 1:
                pcloud_one[j, :] = pointcloud[i, :]
                j = j + 1
            else:
                pcloud_zero[k, :] = pointcloud[i, :]
                k = k + 1
        z_one, z_zero = 0, 0
        for i in range(count_zero):
            z_zero = z_zero + pcloud_zero[i, 2]
        for i in range(count_one):
            z_one = z_one + pcloud_one[i, 2]
        z_zero_av = z_zero / count_zero
        z_one_av = z_one / count_one
        if z_one_av > z_zero_av:
            pointcloud_new = pcloud_zero
        else:
            pointcloud_new = pcloud_one

        point_num = pointcloud_new.shape[0]
        if point_num < 512:
            matrix = np.zeros([512-int(point_num), 3], dtype=np.float)
            pointcloud_new = np.concatenate((pointcloud_new, matrix), axis=0)
            for i in range(point_num):
                pointcloud_new[i+point_num] = pointcloud_new[i]
                if i + point_num == 511:
                    break
            point_num = 512
        indx = np.arange(point_num)
        np.random.shuffle(indx)
        indx = indx[0:setting.sample_num, ...]
        # re-order
        indx = np.sort(indx)
        point_sampled = pointcloud_new[indx, ...]
        result[itr, ...] = point_sampled
    return result


# separate human and background by z value
def GetSampledPcloud_ts(depth_maps, bbox, setting):
    result = np.zeros((depth_maps.shape[0], setting.sample_num, 3))
    for itr in range(depth_maps.shape[0]):
        tmp_depth = depth_maps[itr, ...]
        xmin = bbox[itr, 0]
        xmax = bbox[itr, 1]
        ymin = bbox[itr, 2]
        ymax = bbox[itr, 3]
        pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax)

        # TODO: implement the KMeans sampling methods
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        length = int(len(x))
        count_z = 0
        for i in range(length):
            if z[i] < 3.25:
                count_z = count_z + 1
        pointcloud_new = np.zeros([count_z, 3], dtype=np.float)
        j = 0
        for i in range(length):
            if z[i] < 3.25:
                pointcloud_new[j, :] = pointcloud[i, :]
                j = j + 1

        point_num = pointcloud_new.shape[0]
        if point_num < setting.sample_num:
            matrix = np.zeros([setting.sample_num-int(point_num), 3], dtype=np.float)
            pointcloud_new = np.concatenate((pointcloud_new, matrix), axis=0)
            for i in range(point_num):
                pointcloud_new[i+point_num] = pointcloud_new[i]
                if i + point_num == setting.sample_num-1:
                    break
            point_num = setting.sample_num
        indx = np.arange(point_num)
        np.random.shuffle(indx)
        indx = indx[0:setting.sample_num, ...]
        # re-order
        indx = np.sort(indx)
        point_sampled = pointcloud_new[indx, ...]
        result[itr, ...] = point_sampled
    return result


def GetSampledPcloud_origin(depth_maps, bbox, setting):
    result = np.zeros((depth_maps.shape[0], setting.sample_num, 3))
    for itr in range(depth_maps.shape[0]):
        tmp_depth = depth_maps[itr, ...]
        xmin = bbox[itr, 0]
        xmax = bbox[itr, 1]
        ymin = bbox[itr, 2]
        ymax = bbox[itr, 3]
        pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax)
      
    return pointcloud

def GetBatchPcloud(img, coor):
    c = 0.0035
    batch_size = coor.shape[0]
    p_num = coor.shape[1]
    pcloud = np.zeros([batch_size, p_num, 3])
    pcloud = Imgcoor2Pcloud(img, coor)

    return pcloud

def GetPatchWeight_new(segmentation, bbox_set):
    xlen = int(bbox_set[0, 0, 1] - bbox_set[0, 0, 0])
    ylen = int(bbox_set[0, 0, 3] - bbox_set[0, 0, 2])
    weight = np.zeros((segmentation.shape[0], bbox_set.shape[1], xlen*ylen))

    xmin = bbox_set[:, :, 0].astype(int)
    xmax = bbox_set[:, :, 1].astype(int)
    ymin = bbox_set[:, :, 2].astype(int)
    ymax = bbox_set[:, :, 3].astype(int)
    fir = np.zeros([xmin.shape[0], xmin.shape[1], ylen * xlen], dtype=np.int)
    sec = np.zeros([ymin.shape[0], ymin.shape[1], xlen * ylen], dtype=np.int)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            fir[i, j, :] = np.squeeze(
                np.tile(np.expand_dims(np.arange(xmin[i, j], xmax[i, j]), 1), [1, ylen]).reshape([1, -1]))
            sec[i, j, :] = np.tile(np.arange(ymin[i, j], ymax[i, j]), [xlen])
            weight[i, j, :] = segmentation[i, sec[i,j,:], fir[i,j,:]]
    return weight


def GetEverySampledPcloud(depth_maps, bbox_set, setting):
    result = np.zeros((depth_maps.shape[0], setting.joint_num, setting.sample_num, 3))
    for itr in range(depth_maps.shape[0]):
        for joint in range(setting.joint_num):
            tmp_depth = depth_maps[itr,...]
            xmin = bbox_set[itr, joint, 0]
            xmax = bbox_set[itr, joint, 1]
            ymin = bbox_set[itr, joint, 2]
            ymax = bbox_set[itr, joint, 3]
            pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax)

            # TODO: implement the KMeans sampling methods
            x = pointcloud[:, 0]
            y = pointcloud[:, 1]
            z = pointcloud[:, 2]
            points = [[i, j, k] for i, j, k in zip(x, y, z)]
            points = np.array(points,dtype=np.float32)
          
            pointcloud_new = pointcloud

            point_num = pointcloud_new.shape[0]
            if point_num < setting.sample_num:
                matrix = np.zeros([setting.sample_num-int(point_num), 3], dtype=np.float)
                pointcloud_new = np.concatenate((pointcloud_new, matrix), axis=0)
                for i in range(setting.sample_num):
                    pointcloud_new[i+point_num] = pointcloud_new[i]
                    if i + point_num == setting.sample_num-1:
                        break
                point_num = setting.sample_num
            indx = np.arange(point_num)
            np.random.shuffle(indx)
            indx = indx[0:setting.sample_num, ...]
            # re-order
            indx = np.sort(indx)
            point_sampled = pointcloud_new[indx, ...]
            result[itr, joint, ...] = point_sampled
    return result



def GetEveryPcloud(depth_maps, bbox_set, setting, is_demo='ITOP'):
    pixel_num = (bbox_set[0, 0, 3] - bbox_set[0, 0, 2]) * (bbox_set[0, 0, 1] - bbox_set[0, 0, 0])
    result = np.zeros((depth_maps.shape[0], setting.joint_num, int(pixel_num), 3))
    for i in range(depth_maps.shape[0]):
        for j in range(setting.joint_num):
            tmp_depth = depth_maps[i,...]
            xmin = bbox_set[i, j, 0]
            xmax = bbox_set[i, j, 1]
            ymin = bbox_set[i, j, 2]
            ymax = bbox_set[i, j, 3]
            pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax, is_demo=is_demo)

            result[i, j, ...] = pointcloud
    return result

def GetEveryPcloud_new(depth_maps, segmentation, bbox_set, is_demo='ITOP'):
    result = ConvertDmap_Patch_BBox_new(depth_maps, bbox_set, is_demo=is_demo)
    result_weight = GetPatchWeight_new(segmentation, bbox_set)
    return result, result_weight


def GetRootPcloud(depth_maps, root_box, is_demo='ITOP'):
    pixel_num = (root_box[0, 3] - root_box[0, 2]) * (root_box[0, 1] - root_box[0, 0])
    result = np.zeros((depth_maps.shape[0], int(pixel_num), 3))
    for i in range(depth_maps.shape[0]):
        tmp_depth = depth_maps[i,...]
        xmin = root_box[i, 0]
        xmax = root_box[i, 1]
        ymin = root_box[i, 2]
        ymax = root_box[i, 3]
        pointcloud = ConvertDmap_Patch_BBox(tmp_depth, xmin, xmax, ymin, ymax, is_demo=is_demo)
        result[i, ...] = pointcloud
    return result


def Creatnoise(joint_coor,setting, radius=0):
    joint_coor0=copy.deepcopy(joint_coor)
    randomshift = np.random.uniform(-1 * radius, radius,size=[joint_coor.shape[0],setting.joint_num,2])
    for i in range(joint_coor.shape[0]):
        for j in range(setting.joint_num):
            joint_coor0[i,j,0] = joint_coor0[i,j,0]+randomshift[i,j,0]
            joint_coor0[i,j,1] = joint_coor0[i,j,1]+randomshift[i,j,1]
    return joint_coor0

def farthest_sample(point_cloud, sample_num):
    pc_num = point_cloud.shape[0]
    if pc_num < sample_num:
        sample_idx = np.arange(pc_num)
        sample_idx = [sample_idx, np.random.randint(pc_num, sample_num, size=sample_num - pc_num)]
    else:
        sample_idx = np.zeros([sample_num, 1], dtype=np.int)
        sample_idx[0] = np.random.randint(pc_num)
        cur_sample = np.tile(point_cloud[sample_idx[0], :], (pc_num, 1))
        diff = point_cloud - cur_sample
        min_dist = np.linalg.norm(diff, ord=2, axis=1)

        for cur_sample_idx in range(1, sample_num):
            sample_idx[cur_sample_idx] = min_dist.argmax(axis=0)

            if cur_sample_idx < sample_num:
                for i in range(pc_num):
                    dist = np.linalg.norm(point_cloud[i, :] - point_cloud[sample_idx[cur_sample_idx], :], ord=2)
                    if dist < min_dist[i]:
                        min_dist[i] = dist

    return sample_idx

def calPCKH(pred, ground, threshold=10):
    frames = pred.shape[0]
    njoint = pred.shape[1]
    count = 0
    for i in range(frames):
        error = (pred[i,...]-ground[i,...]) * 100     # m -> cm
        l2_norm = np.linalg.norm(error, ord=2, axis=1)
        for j in range(njoint):
            if l2_norm[j] < threshold:
                count += 1
    return count/(frames * njoint)

def calPCKH_per(pred, ground, threshold=10):
    frames = pred.shape[0]
    njoint = pred.shape[1]
    count = np.zeros((njoint), dtype=np.float32)
    for i in range(frames):
        error = (pred[i,...]-ground[i,...]) * 100     # m -> cm
        l2_norm = np.linalg.norm(error, ord=2, axis=1)
        for j in range(njoint):
            if l2_norm[j] < threshold:
                count[j] += 1
    return count/(frames)

def calPck_time(pred, ground, cur_valid,threshold=10):
    frames = pred.shape[0]
    njoint = pred.shape[1]

    count = np.zeros((njoint), dtype=np.float32)

    total_frame = np.sum(cur_valid)

    cont = np.zeros(shape=[frames])

    for i in range(frames):
        if cur_valid[i]:
            cont[i] = 1
            error = (pred[i, ...] - ground[i, ...]) * 100  # m -> cm
            l2_norm = np.linalg.norm(error, ord=2, axis=1)
            for j in range(njoint):
                if l2_norm[j] < threshold:
                    count[j] += 1

    if total_frame == 0:
        return count, cont
    else:
        return count / total_frame, cont


def depth_visualization(image):
    image=np.squeeze(image)
    img = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    return img



def basic_matrix(translation):
    """ transformation matrix"""
    return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

def adjust_transform_for_image(img, trans_matrix):
    """Adjusts the  transformation matrix based on the image"""
    transform_matrix = copy.deepcopy(trans_matrix)
    height, width, channels = img.shape
    transform_matrix[0:2, 2] *= [width, height]
    center = np.array((0.5 * width, 0.5 * height))
    transform_matrix = np.linalg.multi_dot(
        [basic_matrix(center), transform_matrix, basic_matrix(-center)])
    return transform_matrix

def apply_transform(img,trans_matrix):
    output = cv2.warpAffine(img, trans_matrix[:2, :], dsize=(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return output


def apply(img, trans_matrix,debug=False):
    """use the transformation"""
    img=np.expand_dims(img,axis=-1)
    tmp_matrix = adjust_transform_for_image(img, trans_matrix)
    out_img = apply_transform(img, tmp_matrix)
    if debug == True:
        visu_img=depth_visualization(out_img)
        cv2.imwrite('shiftimg.png',visu_img)
    return out_img

def PcloudNomalization(pcloud, bbox=None):
    '''
    :param pcloud:  float32,(batch_size, njoint, npoint, 3)
    :param bbox: float32, (6, batch_size, njoint)
    :return:

    '''
    # denoising
    idx = np.where(pcloud[..., 2] < 0.1)
    mu = np.mean(pcloud[..., 2])
    pcloud[..., 2][idx] = mu

    # calculate mean and length
    if bbox is None:
        xmin = np.min(pcloud[..., 0], axis=-1)
        xmax = np.max(pcloud[..., 0], axis=-1)
        ymin = np.min(pcloud[..., 1], axis=-1)
        ymax = np.max(pcloud[..., 1], axis=-1)
        zmin = np.min(pcloud[..., 2], axis=-1)
        zmax = np.max(pcloud[..., 2], axis=-1)

        xmean = np.mean(pcloud[..., 0], axis=-1)
        ymean = np.mean(pcloud[..., 1], axis=-1)
        zmean = np.mean(pcloud[..., 2], axis=-1)
        xlength = xmax - xmin
        ylength = ymax - ymin
        zlength = zmax - zmin

        bbox = np.zeros([pcloud.shape[0], pcloud.shape[1], 6])
        bbox[..., 0] = xmean
        bbox[..., 1] = ymean
        bbox[..., 2] = zmean
        bbox[..., 3] = xlength
        bbox[..., 4] = ylength
        bbox[..., 5] = zlength
    else:
        xmean = bbox[..., 0]  #batch_size * joint_num
        ymean = bbox[..., 1]
        zmean = bbox[..., 2]
        xlength = bbox[..., 3]
        ylength = bbox[..., 4]
        zlength = bbox[..., 5]

    new_pcloud = np.ones_like(pcloud)
    new_pcloud[..., 0] = (pcloud[..., 0] - np.tile(np.expand_dims(xmean, axis=2), (1, 1, pcloud.shape[-2]))) / np.tile(
        np.expand_dims(xlength+1e-5, axis=2), (1, 1, pcloud.shape[-2]))
    new_pcloud[..., 1] = (pcloud[..., 1] - np.tile(np.expand_dims(ymean, axis=2), (1, 1, pcloud.shape[-2]))) / np.tile(
        np.expand_dims(ylength+1e-5, axis=2), (1, 1, pcloud.shape[-2]))
    new_pcloud[..., 2] = (pcloud[..., 2] - np.tile(np.expand_dims(zmean, axis=2), (1, 1, pcloud.shape[-2]))) / np.tile(
        np.expand_dims(zlength+1e-5, axis=2), (1, 1, pcloud.shape[-2]))

    return new_pcloud, bbox


def axang2rotm(axis, angle):
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    x = axis[0]
    y = axis[1]
    z = axis[2]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rotm = np.zeros((3, 3))
    rotm[0, 0] = x * xC + ca
    rotm[0, 1] = xyC - zs
    rotm[0, 2] = zxC + ys
    rotm[1, 0] = xyC + zs
    rotm[1, 1] = y * yC + ca
    rotm[2, 0] = zxC - ys
    rotm[2, 1] = yzC + xs
    rotm[2, 2] = z * zC + ca

    return rotm

def cleanBG(pcloud, center, radiu=1.3):
    """
    :param pcloud: Batch_size , joint_num, sample_num, 3
    :param center: Batch_size , 3
    :param radiu:
    :return: pclouds without background
    """
    jn = pcloud.shape[1]
    pcloud_new = np.zeros_like(pcloud)
    center = np.tile(np.expand_dims(center,axis=1),[1,pcloud.shape[2],1])
    for i in range(jn):
        tmp = np.linalg.norm(pcloud[:,i,:,:] - center,axis=2,ord=2)
        ind = np.where(abs(tmp)<radiu)
        ind1 = np.where(abs(tmp)>=radiu)
        pcloud_new[ind[0][:],i,ind[1][:],:] = pcloud[ind[0][:],i,ind[1][:],:]
        # ll = len(ind1[1])
        # ptmp = np.tile(np.expand_dims(pcloud[ind1[0][:],i,0,:],axis=1),[1,ll,1])
        # pcloud_new[ind1[0][:],i,ind1[1][:],:] = ptmp
    return pcloud_new



def TrainingDataAugmentation(pcloud, pose_gt, pose_init, limit=np.pi/6, ratio = 0.4, rot_mat = None):
    is_augment = np.random.rand()

    if is_augment < ratio:
        # rotate the data
        # rotm = axang2rotm([0, 1, 0], limit * (np.random.rand() * 2 - 1))
        rotm = axang2rotm([0, 1, 0], limit)
        pcloud = np.matmul(pcloud, rotm)
        pose_gt = np.matmul(pose_gt, rotm)
        pose_init = np.matmul(pose_init, rotm)
        return pcloud, pose_gt, pose_init
    else:
        if rot_mat is None:
            return pcloud, pose_gt, pose_init
        else:
            for i in range(pcloud.shape[0]):
                pcloud[i, ...] = np.matmul(pcloud[i, ...], rot_mat[i, ...])
                pose_gt[i, ...] = np.matmul(pose_gt[i, ...], rot_mat[i, ...])
                pose_init[i, ...] = np.matmul(pose_init[i, ...], rot_mat[i, ...])
            return pcloud, pose_gt, pose_init


def DeNormalization(pcloud, bbox):
    xmean = bbox[..., 0]
    ymean = bbox[..., 1]
    zmean = bbox[..., 2]
    xlength = bbox[..., 3]
    ylength = bbox[..., 4]
    zlength = bbox[..., 5]


    new_pcloud = np.ones_like(pcloud)

    new_pcloud[..., 0] = pcloud[..., 0] * xlength + xmean
    new_pcloud[..., 1] = pcloud[..., 1] * ylength + ymean
    new_pcloud[..., 2] = pcloud[..., 2] * zlength + zmean

    return  new_pcloud

def next_merge_batch(batch_size, dataset0, dataset1, setting, use_orientation=True):
    batch_size0 = int(batch_size / 2)
    batch_size1 = batch_size - batch_size0

    dmap_0, image_coor_0, world_coor_n_0, pcloud_np_0, bbox_0, o_0 = dataset0.next_normalized_batch(batch_size0, setting, use_orientation=use_orientation)
    dmap_1, image_coor_1, world_coor_n_1, pcloud_np_1, bbox_1, o_1 = dataset1.next_normalized_batch(batch_size1, setting, use_orientation=use_orientation)

    order = np.arange(batch_size)
    np.random.shuffle(order)

    label_0 = np.ones((batch_size0), dtype=np.float32)
    label_1 = np.zeros((batch_size1), dtype=np.float32)
    dmap = np.concatenate([dmap_0, dmap_1], axis=0)
    image_coor = np.concatenate([image_coor_0, image_coor_1], axis=0)
    world_coor_n = np.concatenate([world_coor_n_0, world_coor_n_1], axis=0)
    pcloud_np = np.concatenate([pcloud_np_0, pcloud_np_1], axis=0)
    bbox = np.concatenate([bbox_0, bbox_1], axis=0)
    o = np.concatenate([o_0, o_1], axis=0)
    label = np.concatenate([label_0, label_1], axis=0)

    dmap = dmap[order, ...]
    image_coor = image_coor[order, ...]
    world_coor_n = world_coor_n[order, ...]
    pcloud_np = pcloud_np[order, ...]
    bbox = bbox[order, ...]
    o = o[order, ...]
    label = label[order, ...]

    return dmap, image_coor, world_coor_n, pcloud_np, bbox, o, label

def next_merge_batch_with_init(batch_size, dataset0, dataset1, setting, use_orientation=True):
    batch_size0 = int(batch_size / 2)
    batch_size1 = batch_size - batch_size0

    dmap_0, image_coor_0, world_coor_n_0, pcloud_np_0, init_pose_0, bbox_0, o_0 = dataset0.next_normalized_batch_with_init(batch_size0, setting, use_orientation=use_orientation)
    dmap_1, image_coor_1, world_coor_n_1, pcloud_np_1, init_pose_1, bbox_1, o_1 = dataset1.next_normalized_batch_with_init(batch_size1, setting, use_orientation=use_orientation)

    order = np.arange(batch_size)
    np.random.shuffle(order)

    label_0 = np.ones((batch_size0), dtype=np.float32)
    label_1 = np.zeros((batch_size1), dtype=np.float32)
    dmap = np.concatenate([dmap_0, dmap_1], axis=0)
    image_coor = np.concatenate([image_coor_0, image_coor_1], axis=0)
    world_coor_n = np.concatenate([world_coor_n_0, world_coor_n_1], axis=0)
    pcloud_np = np.concatenate([pcloud_np_0, pcloud_np_1], axis=0)
    init_pose = np.concatenate([init_pose_0, init_pose_1], axis=0)
    bbox = np.concatenate([bbox_0, bbox_1], axis=0)
    o = np.concatenate([o_0, o_1], axis=0)
    label = np.concatenate([label_0, label_1], axis=0)

    dmap = dmap[order, ...]
    image_coor = image_coor[order, ...]
    world_coor_n = world_coor_n[order, ...]
    pcloud_np = pcloud_np[order, ...]
    bbox = bbox[order, ...]
    o = o[order, ...]
    label = label[order, ...]

    return dmap, image_coor, world_coor_n, pcloud_np, init_pose, bbox, o, label

def next_merge_batch_new(batch_size, dataset0, dataset1, setting, use_orientation=True):
    batch_size0 = int(batch_size / 2)
    batch_size1 = batch_size - batch_size0

    dmap_0, image_coor_0, world_coor_n_0, pcloud_np_0, init_pose_0, bbox_0, visible_0 = dataset0.next_normalized_batch_new(batch_size0, setting, use_orientation=use_orientation)
    dmap_1, image_coor_1, world_coor_n_1, pcloud_np_1, init_pose_1, bbox_1, visible_1 = dataset1.next_normalized_batch_new(batch_size1, setting, use_orientation=use_orientation)

    order = np.arange(batch_size)
    np.random.shuffle(order)

    label_0 = np.ones((batch_size0), dtype=np.float32)
    label_1 = np.zeros((batch_size1), dtype=np.float32)
    dmap = np.concatenate([dmap_0, dmap_1], axis=0)
    image_coor = np.concatenate([image_coor_0, image_coor_1], axis=0)
    world_coor_n = np.concatenate([world_coor_n_0, world_coor_n_1], axis=0)
    pcloud_np = np.concatenate([pcloud_np_0, pcloud_np_1], axis=0)
    init_pose = np.concatenate([init_pose_0, init_pose_1], axis=0)
    bbox = np.concatenate([bbox_0, bbox_1], axis=0)
    visible = np.concatenate([visible_0, visible_1], axis=0)
    label = np.concatenate([label_0, label_1], axis=0)

    dmap = dmap[order, ...]
    image_coor = image_coor[order, ...]
    world_coor_n = world_coor_n[order, ...]
    pcloud_np = pcloud_np[order, ...]
    bbox = bbox[order, ...]
    visible = visible[order, ...]
    label = label[order, ...]

    return dmap, image_coor, world_coor_n, pcloud_np, init_pose, bbox, visible, label


def smooth(x,window_len=8,window='flat'):
    if x.ndim != 3:
        raise(ValueError, "smooth only accepts 3 dimension arrays.")
    if x.shape[0] < window_len:
        raise (ValueError, "Input vector needs to be bigger than smooth window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    new_x = np.zeros_like(x,np.float)
    x1 = x[:,:,0]
    x2 = x[:,:,1]
    x3 = x[:,:,2]
    for i in range(x.shape[1]):
        x11 = x1[:,i]
        x22 = x2[:,i]
        x33 = x3[:,i]
        s1 = np.r_[2*x11[0]-x11[window_len-1::-1],x11,2*x11[-1]-x11[-1:-window_len:-1]]
        s2 = np.r_[2 * x22[0] - x22[window_len - 1::-1], x22, 2 * x22[-1] - x22[-1:-window_len:-1]]
        s3 = np.r_[2 * x33[0] - x33[window_len - 1::-1], x33, 2 * x33[-1] - x33[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
        y1=np.convolve(w/w.sum(),s1,mode='same')
        y2=np.convolve(w/w.sum(),s2,mode='same')
        y3 = np.convolve(w / w.sum(), s3, mode='same')
        new_x[:, i, 0] = y1[window_len:-window_len + 1]
        new_x[:, i, 1] = y2[window_len:-window_len + 1]
        new_x[:, i, 2] = y3[window_len:-window_len + 1]
    return new_x


def calBoneStat(boneindex,pred_poses,valid):
    bonelen = np.zeros([int(np.sum(valid)),pred_poses.shape[1]-1])
    cont = 0
    for i in range(len(valid)):
        if valid[i] == 1:
            for j in range(bonelen.shape[1]):
                bonelen[cont,j] = np.linalg.norm(pred_poses[i,boneindex[j][0],:]-pred_poses[i,boneindex[j][1],:],ord=2)
            cont += 1
    return bonelen



