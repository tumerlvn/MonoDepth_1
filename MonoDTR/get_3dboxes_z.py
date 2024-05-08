import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection
from visualDet3D.utils.utils import draw_3D_box
import os
from tqdm import tqdm
from easydict import EasyDict
from typing import Sized, Sequence
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from visualDet3D.evaluator.kitti.evaluate import evaluate
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection
from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
import pickle
from get_chen_number import chen_to_normal



projector = BBox3dProjector().cpu()
backprojector = BackProjection().cpu()

val_chen_num = '000156'
normal_num = chen_to_normal(int(val_chen_num), 'val')

test_num = "000001"

f = test_num + ".png" # normal_num + png
image_dir = "data/KITTI/object/training/image_2/"
calib_dir = "data/KITTI/object/training/calib/"

test_image_dir = "data/KITTI/object/testing/image_2/"
test_calib_dir = "data/KITTI/object/testing/calib/"

image_file = os.path.join(test_image_dir, f) # image_dir
calib_file = test_calib_dir + f.replace('png', 'txt')

val_output_dir = 'workdirs/MonoDTR/output/validation/data/'
test_output_dir = 'workdirs/MonoDTR/output/test/data/'
# predi_file =  os.path.join(val_output_dir, val_chen_num + '.txt')
test_predi_file =  os.path.join(test_output_dir, test_num + '.txt')

image = cv2.imread(image_file)


for line in open(calib_file):
    if 'P2:' in line:
        cam_to_img = line.strip().split(' ')
        cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
        cam_to_img = np.reshape(cam_to_img, (3,4))

cars = []

def convert_z_center_to_meters(P2, x_center, y_center, z_center, l):
    center_homogeneous = np.array([x_center, y_center, 1])
    
    # Compute pseudo-inverse of P2
    P2_pseudo_inverse = np.linalg.pinv(P2)
    
    # Calculate camera coordinate vector
    camera_coordinate_vector = P2_pseudo_inverse @ center_homogeneous
    
    # Extract z-coordinate (depth)
    z_depth = camera_coordinate_vector[2]
    
    # Calculate z-coordinate in meters
    z_meters = z_center * l / z_depth
    
    return z_meters

for line in open(test_predi_file): # 31
        line = line.strip().split(' ')

        print(line)

        dims   = np.asarray([float(number) for number in line[8:11]])
        center = np.asarray([float(number) for number in line[11:14]])
        rot_y  = float(line[3]) + np.arctan(center[0]/center[2])#float(line[14])

        box_3d = []

        for i in [1,-1]:
            for j in [1,-1]:
                for k in [0,1]:
                    point = np.copy(center)
                    point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
                    point[1] = center[1] - k * dims[0]

                    point = np.append(point, 1)
                    point = np.dot(cam_to_img, point)
                    point = point[:2]/point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        for i in range(4):
            point_1_ = box_3d[2*i]
            point_2_ = box_3d[2*i+1]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i+2)%8]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)

        cv2.putText(image, str(center[2]), (point_1_[0], point_1_[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

cv2.imwrite('res.jpg', image)
                