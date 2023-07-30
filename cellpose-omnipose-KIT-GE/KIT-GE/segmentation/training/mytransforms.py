import numpy as np
import random
import scipy
import torch
from imgaug import augmenters as iaa
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.transform import rescale
from torchvision import transforms

from segmentation.utils.utils import min_max_normalization


def augmentors(label_type, min_value, max_value):
    """ Get augmentations for the training process.

    :param label_type: Type of the label images, e.g., 'boundary' or 'distance'.
        :type label_type: str
    :param min_value: Minimum value for the min-max normalization.
        :type min_value: int
    :param max_value: Minimum value for the min-max normalization.
        :type min_value: int
    :return: Dict of augmentations.
    """

    if label_type == 'auto':
        data_transforms = transforms.Compose([CropAndNormalize(),
                                              FlipAuto(p=1.0),
                                              ToTensor(label_type=label_type, min_value=min_value, max_value=max_value)])

    else:
        data_transforms = {'train': transforms.Compose([Flip(p=1.0),
                                                        Contrast(p=0.5),
                                                        Scaling(p=0.25),
                                                        Rotate(p=0.25),
                                                        Blur(p=0.3),
                                                        Noise(p=0.3),
                                                        ToTensor(label_type=label_type,
                                                                 min_value=min_value,
                                                                 max_value=max_value)]),
                           'val': ToTensor(label_type=label_type, min_value=min_value, max_value=max_value)}

    return data_transforms


class Blur(object):
    """ Blur augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            sigma = 1.75 * random.random() + 1.0
            sample['image'] = scipy.ndimage.gaussian_filter(sample['image'], sigma, order=0)
        
        return sample


class Contrast(object):
    """ Contrast augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            img = sample['image']

            h = random.randint(0, 2)

            if h == 0:  # Apply CLAHE or contrast stretching

                img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)
                img = (65535 * img[..., None]).astype(np.uint16)

            elif h == 1:  # Contrast stretching

                p2, p98 = np.percentile(img, (0.2, 99.8))
                img = rescale_intensity(img, in_range=(p2, p98))

            else:  # Apply Contrast and gamma adjustment

                dtype = img.dtype
                img = (img.astype(np.float32) - np.iinfo(dtype).min) / (np.iinfo(dtype).max - np.iinfo(dtype).min)
                contrast_range, gamma_range = (0.65, 1.35), (0.5, 1.5)

                # Contrast
                img_mean, img_min, img_max = img.mean(), img.min(), img.max()
                factor = np.random.uniform(contrast_range[0], contrast_range[1])
                img = (img - img_mean) * factor + img_mean

                # Gamma
                img_mean, img_std, img_min, img_max = img.mean(), img.std(), img.min(), img.max()
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                rnge = img_max - img_min
                img = np.power(((img - img_min) / float(rnge + 1e-7)), gamma) * rnge + img_min

                if random.random() < 0.5:
                    img = 9 * img / 10

                img = np.clip(img, 0, 1)
                img = img * (np.iinfo(dtype).max - np.iinfo(dtype).min) - np.iinfo(dtype).min
                img = img.astype(dtype)

            sample['image'] = img

        return sample


class CropAndNormalize(object):

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        img = sample['image']
        scale = sample['scale']

        if scale < 1:
            img = rescale(np.squeeze(img), scale=scale, order=2, preserve_range=True).astype(img.dtype)
            img = img[..., None]

            if img.shape == (416, 496, 1):
                # Workaround for Flue-C2DL-MSC cell, otherwise batches of both subsets do not match ...
                img = np.pad(img, ((0, 0), (0, 20), (0, 0)), mode='constant')

        img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        img = img.astype(np.uint16)

        if img.shape[0] >= 768:
            crop_height = 768
        elif img.shape[0] >= 512:
            crop_height = 512
        elif img.shape[0] >= 320:
            crop_height = 320
        else:
            crop_height = 256

        if img.shape[1] >= 768:
            crop_width = 768
        elif img.shape[1] >= 512:
            crop_width = 512
        elif img.shape[1] >= 320:
            crop_width = 320
        else:
            crop_width = 256

        seq = iaa.Sequential([iaa.CropToFixedSize(width=crop_width, height=crop_height)])
        img = seq.augment_image(img)

        return {'image': img, 'label': img}


class Flip(object):
    """ Flip and rotation augmentation (label-preserving transformation). Crop needed for non-square images. """

    def __init__(self, p=0.5):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        img = sample['image']

        if random.random() < self.p:

            h = random.randint(0, 7)

            if h == 0:

                pass

            elif h == 1:  # Flip left-right

                sample['image'] = np.flip(img, axis=1).copy()
                if len(sample) == 3:
                    sample['label'] = np.flip(sample['label'], axis=1).copy()
                elif len(sample) == 4:
                    sample['border_label'] = np.flip(sample['border_label'], axis=1).copy()
                    sample['cell_label'] = np.flip(sample['cell_label'], axis=1).copy()

            elif h == 2:  # Flip up-down

                sample['image'] = np.flip(img, axis=0).copy()
                if len(sample) == 3:
                    sample['label'] = np.flip(sample['label'], axis=0).copy()
                elif len(sample) == 4:
                    sample['border_label'] = np.flip(sample['border_label'], axis=0).copy()
                    sample['cell_label'] = np.flip(sample['cell_label'], axis=0).copy()

            elif h == 3:  # Rotate 90°

                sample['image'] = np.rot90(img, axes=(0, 1)).copy()
                if len(sample) == 3:
                    sample['label'] = np.rot90(sample['label'], axes=(0, 1)).copy()
                elif len(sample) == 4:
                    sample['border_label'] = np.rot90(sample['border_label'], axes=(0, 1)).copy()
                    sample['cell_label'] = np.rot90(sample['cell_label'], axes=(0, 1)).copy()

            elif h == 4:  # Rotate 180°

                sample['image'] = np.rot90(img, k=2, axes=(0, 1)).copy()
                if len(sample) == 3:
                    sample['label'] = np.rot90(sample['label'], k=2, axes=(0, 1)).copy()
                elif len(sample) == 4:
                    sample['border_label'] = np.rot90(sample['border_label'], k=2, axes=(0, 1)).copy()
                    sample['cell_label'] = np.rot90(sample['cell_label'], k=2, axes=(0, 1)).copy()

            elif h == 5:  # Rotate 270°

                sample['image'] = np.rot90(img, k=3, axes=(0, 1)).copy()
                if len(sample) == 3:
                    sample['label'] = np.rot90(sample['label'], k=3, axes=(0, 1)).copy()
                elif len(sample) == 4:
                    sample['border_label'] = np.rot90(sample['border_label'], k=3, axes=(0, 1)).copy()
                    sample['cell_label'] = np.rot90(sample['cell_label'], k=3, axes=(0, 1)).copy()

            elif h == 6:  # Flip left-right + rotate 90°

                img = np.flip(img, axis=1).copy()
                sample['image'] = np.rot90(img, axes=(0, 1)).copy()

                if len(sample) == 3:
                    label_img = np.flip(sample['label'], axis=1).copy()
                    sample['label'] = np.rot90(label_img, k=1, axes=(0, 1)).copy()
                elif len(sample) == 4:
                    border_label = np.flip(sample['border_label'], axis=1).copy()
                    cell_label = np.flip(sample['cell_label'], axis=1).copy()
                    sample['border_label'] = np.rot90(border_label, k=1, axes=(0, 1)).copy()
                    sample['cell_label'] = np.rot90(cell_label, k=1, axes=(0, 1)).copy()

            elif h == 7:  # Flip up-down + rotate 90°

                img = np.flip(img, axis=0).copy()
                sample['image'] = np.rot90(img, axes=(0, 1)).copy()

                if len(sample) == 3:
                    label_img = np.flip(sample['label'], axis=0).copy()
                    sample['label'] = np.rot90(label_img, k=1, axes=(0, 1)).copy()
                elif len(sample) == 4:
                    border_label = np.flip(sample['border_label'], axis=0).copy()
                    cell_label = np.flip(sample['cell_label'], axis=0).copy()
                    sample['border_label'] = np.rot90(border_label, k=1, axes=(0, 1)).copy()
                    sample['cell_label'] = np.rot90(cell_label, k=1, axes=(0, 1)).copy()

        return sample


class FlipAuto(object):
    """ Flip and rotation augmentation for auto-encoder pre-training. """

    def __init__(self, p=0.5):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        img = sample['image']

        if random.random() < self.p:

            h = random.randint(0, 3)

            if h == 0:
                pass
            elif h == 1:  # Flip left-right
                img = np.flip(img, axis=1).copy()
            elif h == 2:  # Flip up-down
                img = np.flip(img, axis=0).copy()
            elif h == 3:  # Rotate 180°
                img = np.rot90(img, k=2, axes=(0, 1)).copy()

        return {'image': img, 'label': img}


class Noise(object):
    """ Gaussian noise augmentation """

    def __init__(self, p=0.25):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            # Add noise with sigma 1-6% of image maximum
            sigma = random.randint(1, 6) / 100 * np.max(sample['image'])

            # Add noise to selected images
            seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=sigma, per_channel=False)])
            sample['image'] = seq.augment_image(sample['image'])

        return sample

 
class Rotate(object):
    """ Rotation augmentation (label-changing augmentation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        self.angle = (-45, 45)
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            angle = random.uniform(self.angle[0], self.angle[1])
                
            seq1 = iaa.Sequential([iaa.Affine(rotate=angle)]).to_deterministic()
            seq2 = iaa.Sequential([iaa.Affine(rotate=angle, order=0)]).to_deterministic()
            sample['image'] = seq1.augment_image(sample['image'])

            if len(sample) == 3:
                if sample['label'].dtype == np.uint8:
                    sample['label'] = seq2.augment_image(sample['label'])
                else:
                    sample['label'] = seq1.augment_image(sample['label'])

            elif len(sample) == 4:

                if sample['border_label'].dtype == np.uint8:
                    sample['border_label'] = seq2.augment_image(sample['border_label'])
                else:
                    sample['border_label'] = seq1.augment_image(sample['border_label'])

                if sample['cell_label'].dtype == np.uint8:
                    sample['cell_label'] = seq2.augment_image(sample['cell_label'])
                else:
                    sample['cell_label'] = seq1.augment_image(sample['cell_label'])
            else:
                raise Exception('Unsupported sample format.')
         
        return sample


class Scaling(object):
    """ Scaling augmentation (label-changing transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        self.scale = (0.8, 1.20)

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            scale1 = random.uniform(self.scale[0], self.scale[1])
            scale2 = random.uniform(self.scale[0], self.scale[1])

            seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2})])
            seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2}, order=0)])
            sample['image'] = seq1.augment_image(sample['image'])

            if len(sample) == 3:
                if sample['label'].dtype == np.uint8:
                    sample['label'] = seq2.augment_image(sample['label'])
                else:
                    sample['label'] = seq1.augment_image(sample['label']).copy()
            elif len(sample) == 4:
                if sample['border_label'].dtype == np.uint8:
                    sample['border_label'] = seq2.augment_image(sample['border_label'])
                else:
                    sample['border_label'] = seq1.augment_image(sample['border_label'])

                if sample['cell_label'].dtype == np.uint8:
                    sample['cell_label'] = seq2.augment_image(sample['cell_label'])
                else:
                    sample['cell_label'] = seq1.augment_image(sample['cell_label'])
            else:
                raise Exception('Unsupported sample format.')

        return sample

  
class ToTensor(object):
    """ Convert image and label image to Torch tensors """
    
    def __init__(self, label_type, min_value, max_value):
        """

        :param min_value: Minimum value for the normalization. All values below this value are clipped
            :type min_value: int
        :param max_value: Maximum value for the normalization. All values above this value are clipped.
            :type max_value: int
        """
        self.min_value = min_value
        self.max_value = max_value
        self.label_type = label_type
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        # Normalize image
        sample['image'] = min_max_normalization(sample['image'], min_value=self.min_value, max_value=self.max_value)

        # Swap axes from (H, W, Channels) to (Channels, H, W)
        for key in sample:
            if key != 'id':
                sample[key] = np.transpose(sample[key], (2, 0, 1))

        img = torch.from_numpy(sample['image']).to(torch.float)

        if self.label_type == 'auto':
            label = sample['label']
            label = label.astype(np.float32) / 65535  # normalize to [0, 1]
            label = torch.from_numpy(label).to(torch.float)

            return img, label

        # loss function (l1loss/l2loss) needs float tensor with shape [batch, channels, height, width]
        cell_label = torch.from_numpy(sample['cell_label']).to(torch.float)
        border_label = torch.from_numpy(sample['border_label']).to(torch.float)

        return img, border_label, cell_label
