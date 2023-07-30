import numpy as np
import tifffile as tiff
import torch

from skimage.exposure import equalize_adapthist
from skimage.transform import rescale
from torch.utils.data import Dataset
from torchvision import transforms

from segmentation.utils.utils import zero_pad_model_input


class CTCDataSet(Dataset):
    """ Pytorch data set for Cell Tracking Challenge data. """

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: Directory with the Cell Tracking Challenge images to predict (e.g., t001.tif)
        :param transform:
        """

        self.img_ids = sorted(data_dir.glob('t*.tif'))
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        sample = {'image': img,
                  'id': img_id.stem}

        sample = self.transform(sample)

        return sample

class CTCDataSet_test(Dataset):
    """ Pytorch data set for Cell Tracking Challenge data. """

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: Directory with the Cell Tracking Challenge images to predict (e.g., t001.tif)
        :param transform:
        """

        self.img_ids = sorted(data_dir.glob('*.tif'))
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        sample = {'image': img,
                  'id': img_id.stem}

        sample = self.transform(sample)

        return sample

def pre_processing_transforms(apply_clahe, scale_factor):
    """ Get transforms for the CTC data set.

    :param apply_clahe: apply CLAHE.
        :type apply_clahe: bool
    :param scale_factor: Downscaling factor <= 1.
        :type scale_factor: float

    :return: transforms
    """

    data_transforms = transforms.Compose([ContrastEnhancement(apply_clahe),
                                          Normalization(),
                                          Scaling(scale_factor),
                                          Padding(),
                                          ToTensor()])

    return data_transforms


class ContrastEnhancement(object):

    def __init__(self, apply_clahe):
        self.apply_clahe = apply_clahe

    def __call__(self, sample):

        if self.apply_clahe:
            img = sample['image']
            img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)
            img = (65535 * img).astype(np.uint16)
            sample['image'] = img

        return sample


class Normalization(object):

    def __call__(self, sample):

        img = sample['image']

        img = 2 * (img.astype(np.float32) - img.min()) / (img.max() - img.min()) - 1

        sample['image'] = img

        return sample


class Padding(object):

    def __call__(self, sample):

        img = sample['image']
        img, pads = zero_pad_model_input(img=img, pad_val=np.min(img))
        sample['image'] = img
        sample['pads'] = pads

        return sample


class Scaling(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):

        img = sample['image']
        sample['original_size'] = img.shape

        if self.scale < 1:
            if len(img.shape) == 3:
                img = rescale(img, (1, self.scale, self.scale), order=2, preserve_range=True).astype(img.dtype)
            else:
                img = rescale(img, (self.scale, self.scale), order=2, preserve_range=True).astype(img.dtype)
            sample['image'] = img

        return sample


class ToTensor(object):
    """ Convert image and label image to Torch tensors """

    def __call__(self, sample):

        img = sample['image']

        if len(img.shape) == 2:
            img = img[None, :, :]

        img = torch.from_numpy(img).to(torch.float)

        return img, sample['id'], sample['pads'], sample['original_size']
