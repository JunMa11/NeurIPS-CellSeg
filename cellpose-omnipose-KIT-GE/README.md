# Evaluate Cellpose, Omnipose, and KIT-GE ([CTC](http://celltrackingchallenge.net/) top solution in the segmentation benchmark)



## Preprocessing

Download training data and set proper path

Preprocess dataset with

```shell
python pre_img_for_cellpose_omnipose
```

> This challenge focused on universal segmentation methods, while Cellpose and Omnipose required to manually specify the image channel to be segmented. All images were converted to grey images. 


## Segmentation with Cellpose

Install Cellpose https://github.com/MouseLand/cellpose

```bash

python test_cellpose_cyto2_grey

```

## Segmentation with Cellpose2

Train the model on the challenge dataset

```bash

python -m cellpose --train --use_gpu --dir path_to_pre-train --chan 0 --chan2 0 --n_epochs 500 --pretrained_model cyto2 --batch_size 32 --dir_above --save_each --verbose --img_filter '_img' --mask_filter '_masks'

```


Download the trained model: https://drive.google.com/drive/folders/1dpLA1XFAACuwk5V6PGJLSQFk98REQ721?usp=sharing

Inference

```bash

python test_cellpose2_grey

```


## Segmentation with Omnipose

Install Omnipose https://github.com/kevinjohncutler/omnipose
```bash

python test_omnipose_cyto2_grey

```


## Segmentation with KIT-GE

Install the package: https://github.com/TimScherr/KIT-GE-3-Cell-Segmentation-for-CTC

Follow the [CTC dataset](http://celltrackingchallenge.net/2d-datasets/) format to rename this challenge dataset. Here is the expected folder structure.

```bash
BFDP-NeurIPS-Cell # B:Brightfield; F:Fluorescent; D:DIC; P:Phase-contrast
----01
--------t0001.tif
--------t0002.tif
--------t0003.tif
--------tXXXX.tif
--------t1000.tif
----01_GT
--------SEG
------------man_seg0001.tif
------------man_seg0002.tif
------------man_seg0003.tif
------------man_segXXXX.tif
------------man_seg1000.tif
```

Train:

```bash

python train.py
```

Download the trained model: https://drive.google.com/drive/folders/1cwrl4fhwYSga9hetG9Mt-emYzj4CR5ja?usp=sharing

Inference:

```bash
python infer_neurips.py

```