#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cellpose import models, io
import os
join = os.path.join
import numpy as np
from skimage.io import imshow, imsave
from skimage import segmentation
from tqdm import tqdm
import tifffile as tif
import warnings
warnings.filterwarnings("ignore")

# DEFINE CELLPOSE MODEL
model_path = 'path to/cellpose2.0/model/model.501776_epoch_499'
model = models.CellposeModel(gpu=True, pretrained_model = model_path, net_avg=False)
print('model info:', model.gpu, model.pretrained_model)
print('start predicting....')
#%%
img_path = 'path to pre-val/test'
seg_path = 'path to cp2-retrain-seg'
names = sorted(os.listdir(img_path))
os.makedirs(seg_path, exist_ok=True)
chan = [0, 0]

for i, name in enumerate(tqdm(names)):
    save_name = name.split('.')[0]+'_label.tiff'
    if not os.path.isfile(join(seg_path, save_name)):
        img = tif.imread(join(img_path, name))
        masks, flows, styles = model.eval(img, diameter=None, channels=chan, net_avg=False, progress=True)
        tif.imwrite(join(seg_path, save_name), masks, compression='zlib')

