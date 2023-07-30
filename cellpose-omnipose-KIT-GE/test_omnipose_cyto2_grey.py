#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cellpose_omni import models
import os
join = os.path.join
import numpy as np
from tqdm import tqdm
import tifffile as tif
import skimage.io
import warnings
warnings.filterwarnings("ignore")
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print('start predicting....')
#%%
img_path = 'path to pre-val/test'
seg_path = 'path to cp-default-seg'
os.makedirs(seg_path, exist_ok=True)
names = sorted(os.listdir(img_path))

model = models.CellposeModel(gpu=True, model_type='cyto2')
for name in tqdm(names):
    imgs = [tif.imread(join(img_path, name))]
    masks, flows, styles = model.eval(imgs, diameter=None, channels=[0, 0], omni=True)
    save_name = name.split('.')[0]+'_label.tiff'
    tif.imwrite(join(seg_path, save_name), masks[0], compression='zlib')











