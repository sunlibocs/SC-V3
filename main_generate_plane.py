import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import stats
import skimage.measure
import json
import os
from imageio import imread
import torch
from Surface_normal import *
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import cmath
CROP = 16
Height = 480
Width = 640
VALIDNUM_THR = 480 * 640 * 0.02
label2color_dict_0 = [
    [255, 248, 220],  # cornsilk
    [101, 149, 237],  # cornflowerblue
    [102, 205, 170],  # mediumAquamarine
    [205, 133, 163],  # peru
    [160, 132, 240],  # purple
    [255, 164, 164],  # brown1
    [139, 169, 119],  # Chocolate4
    [200, 164, 164],
    [139, 200, 119],
]
label2color_dict = [
    [0, 0, 0],
    [255, 248, 220],  # cornsilk
    [101, 149, 237],  # cornflowerblue
    [102, 205, 170],  # mediumAquamarine
    [205, 133, 163],  # peru
    [160, 132, 240],  # purple
    [255, 164, 164],  # brown1
    [139, 169, 119],  # Chocolate4

    [200, 164, 164],
    [139, 200, 119],
]
for i0 in range(1, 4):
    for i1 in range(0, 4):
        for i2 in range(0, 4):
            for j in range(len(label2color_dict_0)):
                elem = label2color_dict_0[j].copy()
                elem[0] = elem[0] - i0 * 25
                elem[1] = elem[1] - i1 * 25
                elem[2] = elem[2] - i2 * 25
                label2color_dict.append(elem)


def extract_superpixel(filename, index):
    scales = [1]
    markers = [400]
    image = cv2.imread(filename)
    image = image[CROP:-CROP, CROP:-CROP, :]
    segments = []
    for s, m in zip(scales, markers):
        #image = cv2.resize(image, (384 // s, 288 // s))
        image = img_as_float(image)
        segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        segments.append(segment)

    return segments[0].astype(np.int16)

def normal2plane(filenames, save_path):
    segments = {i: extract_superpixel(filenames, 0) for i in [0]}
    seg_rs = segments[0].copy()

    #seg_rs = cv2.resize(seg_rs, (608, 448), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
    seg_full = np.zeros((Height, Width), dtype=np.int16)
    visual_anno = np.zeros((seg_full.shape[0], seg_full.shape[1], 3), dtype=np.uint8)
    seg_full[CROP:-CROP, CROP:-CROP] = seg_rs
    # cv2.imwrite(save_path.replace('.png', 'plane_grey.png'), seg_full)
    for i in range(1, seg_full.max()+1):  ##TODO
        mask_invalid = seg_full == i
        if len(seg_full[mask_invalid]) < VALIDNUM_THR:
            seg_full[mask_invalid] = 0

    cv2.imwrite(save_path, seg_full)
    if seg_full.max() >= 442:##TODO
        mask = seg_full>=442
        seg_full[mask] = seg_full[mask] % 442
    for i in range(seg_full.max()+1):
        mask = seg_full == i
        visual_anno[mask] = label2color_dict[i]

    cv2.imwrite(save_path.replace('.png', '.jpg'), visual_anno)
    return seg_full


def rgb2plane(filenames, save_path):
    segments = {i: extract_superpixel(filenames, 0) for i in [0]}
    seg_rs = segments[0].copy()

    #seg_rs = cv2.resize(seg_rs, (608, 448), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
    seg_full = np.zeros((Height, Width), dtype=np.int16)
    visual_anno = np.zeros((seg_full.shape[0], seg_full.shape[1], 3), dtype=np.uint8)
    seg_full[CROP:-CROP, CROP:-CROP] = seg_rs
    # cv2.imwrite(save_path.replace('.png', 'rgb_plane_grey.png'), seg_full)
    for i in range(1, seg_full.max()+1):  ##TODO
        mask_invalid = seg_full == i
        if len(seg_full[mask_invalid]) < VALIDNUM_THR:
            seg_full[mask_invalid] = 0
    cv2.imwrite(save_path, seg_full)
    if seg_full.max() >= 442:##TODO
        mask = seg_full>=442
        seg_full[mask] = seg_full[mask] % 442
    for i in range(seg_full.max()+1):
        mask = seg_full == i
        visual_anno[mask] = label2color_dict[i]

    cv2.imwrite(save_path.replace('.png', '.jpg'), visual_anno)
    return seg_full


def generate_plane():
    results=[]
    #dataset_path = './nyu/'
    dataset_path = '/home/libo/Research/depth_data/nyu/training/'

    depth_name = '/pseudo_depth2'
    with open(dataset_path + 'train.txt','r') as f:
        for line in f:
            temp_name = line.strip('\n')
            results.append(dataset_path + temp_name + depth_name)
            path_normal_plane = dataset_path + temp_name + '/pseudo_normal_plane'
            path_rgb_plane = dataset_path + temp_name + '/pseudo_rgb_plane'
            path_normal = dataset_path + temp_name + '/pseudo_normal'

            folder_normal = os.path.exists(path_normal)
            folder_normal_plane = os.path.exists(path_normal_plane)
            folder_rgb_plane = os.path.exists(path_rgb_plane)
            if not folder_normal:
                os.makedirs(path_normal)
            if not folder_normal_plane:
                os.makedirs(path_normal_plane)
            if not folder_rgb_plane:
                os.makedirs(path_rgb_plane)
    countN = 0
    for item in results:
        all_depthImgs = os.listdir(item)
        for item2 in all_depthImgs:
            pathD = item + '/' + item2
            depth = imread(pathD).astype(np.float32)
            img_path = pathD.replace(depth_name, '').replace('png', 'jpg')
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda()
            fx = 519.
            surfaceNormal = surface_normal_from_depth(depth_tensor, fx, depth_tensor>0.0)
            normal = surfaceNormal[0].cpu().permute(1,2,0).numpy()
            import matplotlib.pyplot as plt
            normal_path = pathD.replace(depth_name, '/pseudo_normal')
            normal_data = (normal-normal.min())/(normal-normal.min()).max()
            plt.imsave(normal_path, normal_data)

            normal2plane(normal_path, pathD.replace(depth_name, '/pseudo_normal_plane'))
            rgb2plane(img_path, pathD.replace(depth_name, '/pseudo_rgb_plane'))

            countN += 1
            print('processing {} current N = {}'.format(img_path, countN))



if __name__ == '__main__':
    generate_plane()
