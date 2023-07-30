import json
import numpy as np
import tifffile as tiff
from random import randint, shuffle
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, data_dir, train_dir, transform=lambda x: x):
        """

        :param data_dir: Directory containing the Cell Tracking Challenge Data.
            :type data_dir: pathlib Path object.
        :param train_dir: Directory containing the to the cell type belonging train data (needed to get the).
            :type train_dir: pathlib Path object.
        :param transform: transforms/augmentations.
            :type transform:
        :return sample (image, image, scale).
        """

        self.img_ids = []

        img_ids_01 = sorted((data_dir / '01').glob('*.tif'))
        if len(img_ids_01) > 1500:
            img_ids_01 = img_ids_01[1500:]
        elif len(img_ids_01) > 1000:
            img_ids_01 = img_ids_01[1000:]
        while len(img_ids_01) > 75:
            img_ids_01 = img_ids_01[::5]
        if len(img_ids_01) > 15:
            shuffle(img_ids_01)
            img_ids_01 = img_ids_01[:15]

        img_ids_02 = sorted((data_dir / '02').glob('*.tif'))
        if len(img_ids_02) > 1500:
            img_ids_02 = img_ids_02[1500:]
        elif len(img_ids_02) > 1000:
            img_ids_02 = img_ids_02[1000:]
        while len(img_ids_02) > 75:
            img_ids_02 = img_ids_02[::5]
        if len(img_ids_02) > 15:
            shuffle(img_ids_02)
            img_ids_02 = img_ids_02[:15]

        if tiff.imread(str(img_ids_01[0])).shape == tiff.imread(str(img_ids_02[0])).shape:
            self.img_ids = img_ids_01 + img_ids_02
        else:
            print('Subsets 01 and 02 have different sizes')
            self.img_ids = img_ids_01  # just use one subset

        with open(train_dir / 'info.json') as f:
            self.scale = json.load(f)['scale']

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        if len(img.shape) == 2:
            img = img[..., None]
        else:
            img_mean, img_std = np.mean(img), np.std(img)
            i = randint(0, img.shape[0]-1)
            h = 0
            while np.mean(img[i]) < img_mean and h <= 10:
                i = randint(0, img.shape[0]-1)
                h += 1
            img = img[i, :, :, None]

        sample = {'image': img,
                  'label': img,
                  'scale': self.scale}

        sample = self.transform(sample)

        return sample
