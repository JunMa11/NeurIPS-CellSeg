#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cellpose import models, io
import os
join = os.path.join
import numpy as np
from tqdm import tqdm
import tifffile as tif
import warnings
warnings.filterwarnings("ignore")
import time

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto2', net_avg=False)


print('start predicting....')
#%%
img_path = 'path to pre-val/test'
seg_path = 'path to cp-default-seg'
os.makedirs(seg_path, exist_ok=True)

names = sorted(os.listdir(img_path))
# The first channel is the channel you want to segment. 
# The second channel is an optional channel that is helpful in models trained with images with a nucleus channel. 
chan = [0, 0]
time_costs = []
for i, name in enumerate(tqdm(names)):
    save_name = name.split('.')[0]+'_label.tiff'
    img = tif.imread(join(img_path, name))
    masks = model.eval(img, diameter=None, channels=chan, net_avg=False, progress=True)
    tif.imwrite(join(seg_path, save_name), masks[0], compression='zlib')












