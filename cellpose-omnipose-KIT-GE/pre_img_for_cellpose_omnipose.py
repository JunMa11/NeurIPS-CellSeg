#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 16 13:12:04 2023

@author: jma
"""

import os
join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure, color
import numpy as np
import tifffile as tif
from tqdm import tqdm


tr_img_path = 'path to training images' 
gt_path =  'path to training GT'
target_path = 'path to pre-train'
os.makedirs(target_path, exist_ok=True)

img_names = sorted(os.listdir(tr_img_path))
gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]

for img_name, gt_name in zip(tqdm(img_names), gt_names):
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(join(tr_img_path, img_name))
    else:
        img_data = io.imread(join(tr_img_path, img_name))
    gt_data = tif.imread(join(gt_path, gt_name))
    cellpose_dataformat = np.zeros((img_data.shape[0], img_data.shape[1]), dtype=img_data.dtype)

    if len(img_data.shape) == 2:
        cellpose_dataformat = img_data
    else: # three-channel images, convert to grey images, put it in green channel as cellpose required
        img_max = np.max(img_data)
        cellpose_dataformat = color.rgb2gray(img_data)*img_max

    basename = img_name.split('.')[0]
    tif.imwrite(join(target_path, basename+'_img.tif'), cellpose_dataformat)
    tif.imwrite(join(target_path, basename+'_masks.tif'), gt_data.astype(np.uint16))

ts_img_path = 'path to val/test images'
ts_pre_path = 'path to pre-val/test'
os.makedirs(ts_pre_path, exist_ok=True)
names = sorted(os.listdir(ts_img_path))
print('Processing test images ...')
for img_name in tqdm(names):
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(join(ts_img_path, img_name))
    else:
        img_data = io.imread(join(ts_img_path, img_name))
    cellpose_dataformat = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=img_data.dtype)
    
    if len(img_data.shape) == 2:
        cellpose_dataformat = img_data
    else: # three-channel images, convert to grey images, put it in green channel as cellpose required
        img_max = np.max(img_data)
        cellpose_dataformat = color.rgb2gray(img_data)*img_max

    basename = img_name.split('.')[0]
    tif.imwrite(join(ts_pre_path, basename+'.tif'), cellpose_dataformat)


















