import hashlib
import json
import math
import numpy as np
import os
import tifffile as tiff

from pathlib import Path
from random import shuffle, random
from scipy.ndimage import gaussian_filter
import shutil
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening
from skimage.transform import rescale

from segmentation.training.train_data_representations import distance_label_2d
from segmentation.utils.utils import get_nucleus_ids


def adjust_dimensions(crop_size, *imgs):
    """ Adjust dimensions so that only 'complete' crops are generated.

    :param crop_size: Size of the (square) crops.
        :type crop_size: int
    :param imgs: Images to adjust the dimensions.
        :type imgs:
    :return: img with adjusted dimension.
    """

    img_adj = []

    # Add pseudo color channels
    for img in imgs:
        img = np.expand_dims(img, axis=-1)

        pads = []
        for i in range(2):
            if img.shape[i] < crop_size:
                pads.append((0, crop_size - (img.shape[i] % crop_size)))
            elif img.shape[i] == crop_size:
                pads.append((0, 0))
            else:
                if (img.shape[i] % crop_size) < 0.075 * img.shape[i]:
                    idx_start = (img.shape[i] % crop_size) // 2
                    idx_end = img.shape[i] - ((img.shape[i] % crop_size) - idx_start)
                    if i == 0:
                        img = img[idx_start:idx_end, ...]
                    else:
                        img = img[:, idx_start:idx_end, ...]
                    pads.append((0, 0))
                else:
                    pads.append((0, crop_size - (img.shape[i] % crop_size)))

        img = np.pad(img, (pads[0], pads[1], (0, 0)), mode='constant')

        img_adj.append(img)

    return img_adj


def close_mask(mask, apply_opening=False, kernel_closing=np.ones((10, 10)), kernel_opening=np.ones((10, 10))):
    """ Morphological closing of STs.

    :param mask: Segmentation mask (gold truth or silver truth).
        :type mask: numpy array.
    :param apply_opening: Apply opening or not (basically needed to correct slices from 3D silver truth).
        :type apply_opening: bool.
    :param kernel_closing: Kernel for closing.
        :type kernel_closing: numpy array.
    :param kernel_opening: Kernel for opening.
        :type kernel_opening: numpy array.
    :return: Closed (and opened) mask.
    """

    # Get nucleus ids and close/open the nuclei separately
    nucleus_ids = get_nucleus_ids(mask)
    hlabel = np.zeros(shape=mask.shape, dtype=mask.dtype)
    for nucleus_id in nucleus_ids:
        nucleus = mask == nucleus_id
        # Close nucleus gaps
        nucleus = binary_closing(nucleus, kernel_closing)
        # Remove small single not connected pixels
        if apply_opening:
            nucleus = binary_opening(nucleus, kernel_opening)
        hlabel[nucleus] = nucleus_id.astype(mask.dtype)

    return hlabel


def copy_train_data(source_path, target_path, idx):
    """  Copy generated training data crops.

    :param source_path: Directory containing the training data crops.
        :type source_path: pathlib Path object
    :param target_path: Directory to copy the training data crops into.
        :type target_path: pathlib Path Object
    :param idx: path/id of the training data crops to copy.
        :type idx: pathlib Path Object
    :return: None
    """
    shutil.copyfile(str(source_path / "img_{}.tif".format(idx)), str(target_path / "img_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)), str(target_path / "dist_cell_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "dist_neighbor_{}.tif".format(idx)), str(target_path / "dist_neighbor_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)), str(target_path / "mask_{}.tif".format(idx)))
    return


def copy_train_set(source_path, target_path, mode='GT'):
    """  Copy generated training data sets (train and val).

    :param source_path: Directory containing the training data sets.
        :type source_path: pathlib Path object.
    :param target_path: Directory to copy the training data sets into.
        :type target_path: pathlib Path Object
    :param mode: 'GT' deletes possibly existing train and val directories.
        :type mode: str
    :return: None
    """

    if mode == 'GT':
        os.rmdir(str(target_path / 'train'))
        os.rmdir(str(target_path / 'val'))
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val'))
    else:
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train_st'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val_st'))


def downscale(img, scale, order=2, aa=None):
    """ Downscale image and segmentation ground truth.

    :param img: Image to downscale
        :type img:
    :param scale: Scale for downscaling.
        :type scale: float
    :param order: Order of the polynom used.
        :type order: int
    :param aa: apply anti-aliasing (not recommended for the masks).
        :type aa: bool
    :return: downscale images.
    """
    if len(img.shape) == 3:
        scale_img = (1, scale, scale)
    else:
        scale_img = (scale, scale)
    img = rescale(img, scale=scale_img, order=order, anti_aliasing=aa, preserve_range=True).astype(img.dtype)

    return img


def foi_correction_train(cell_type, mode, *imgs):
    """ Field of interest correction (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    For the GTs, the foi correction differs a bit for some cell types since some GT training data sets were already
    fixed before we have seen the need for the foi correction. However, this should not affect the results
    but affects the cropping and fully annotated crop selection and therefore is needed for reproducibility of our sets.

    :param cell_type: Cell type / dataset (needed for filename).
        :type cell_type: str
    :param mode: 'GT' gold truth, 'ST' silver truth.
        :type mode: str
    :param imgs: Images to adjust the dimensions.
        :type imgs:
    :return: foi corrected images.
    """

    if mode == 'GT':
        if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
            E = 50
        elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
            E = 25
        else:
            E = 0
    else:
        if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373', 'Fluo-C3DH-H157',
                         'Fluo-N3DH-CHO']:
            E = 50
        elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
            E = 25
        else:
            E = 0

    img_corr = []

    for img in imgs:
        if len(img.shape) == 2:
            img_corr.append(img[E:img.shape[0] - E, E:img.shape[1] - E])
        else:
            img_corr.append(img[:, E:img.shape[1] - E, E:img.shape[2] - E])

    return img_corr


def generate_data(img, mask, tra_gt, td_settings, cell_type, mode, subset, frame, path, crop_idx=0, slice_idx=None):
    """ Calculate cell and neighbor distances and create crops.

    :param img: Image.
        :type img: numpy array
    :param mask: (Segmentation) Mask / label image (intensity coded).
        :type mask: numpy array
    :param tra_gt: Tracking ground truth (needed to evaluate if all cells are annotated).
        :type tra_gt: numpy array
    :param td_settings: training data creation settings (search radius, crop size ...).
        :type td_settings: dict
    :param cell_type: Cell type / dataset (needed for filename).
        :type cell_type: str
    :param mode: 'GT' gold truth, 'ST' silver truth.
        :type mode: str
    :param subset: Subset from which the image and segmentation mask come from (needed for filename) ('01', '02').
        :type subset: str
    :param frame: Frame of the time series (needed for filename).
        :type frame: str
    :param path: Path to save the generated crops.
        :type path: Pathlib Path object
    :crop_idx: Index to count generated crops and break the data generation if the maximum ammount is reached (for STs)
        :type crop_idx: int
    :param slice_idx: Slice index (for 3D data).
        :type slice_idx: int
    :return: None
    """

    # Calculate train data representations
    cell_dist, neighbor_dist = distance_label_2d(label=mask,
                                                 cell_radius=int(np.ceil(0.5 * td_settings['max_mal'])),
                                                 neighbor_radius=td_settings['search_radius'])

    # Adjust image dimensions for appropriate cropping
    img, mask, cell_dist, neighbor_dist, tra_gt = adjust_dimensions(td_settings['crop_size'], img, mask, cell_dist,
                                                                    neighbor_dist, tra_gt)

    # Cropping
    nx, ny = math.floor(img.shape[1] / td_settings['crop_size']), math.floor(img.shape[0] / td_settings['crop_size'])
    for y in range(ny):
        for x in range(nx):

            # Crop
            img_crop, mask_crop, cell_dist_crop, neighbor_dist_crop, tra_gt_crop = get_crop(x, y, td_settings['crop_size'],
                                                                                            img, mask, cell_dist,
                                                                                            neighbor_dist, tra_gt)
            # Get crop name
            if slice_idx is not None:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}_{:02d}.tif'.format(cell_type, mode, subset, frame, slice_idx, y, x)
            else:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}.tif'.format(cell_type, mode, subset, frame, y, x)

            # Check cell number TRA/SEG
            tr_ids, mask_ids = get_nucleus_ids(tra_gt_crop), get_nucleus_ids(mask_crop)
            if np.sum(mask_crop[10:-10, 10:-10, 0] > 0) < td_settings['min_area']:  # only cell parts / no cell
                continue
            if len(mask_ids) == 1:  # neighbor may be cut from crop --> set dist to 0
                neighbor_dist_crop = np.zeros_like(neighbor_dist_crop)
            if np.sum(img_crop == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):  # almost background
                # For (e.g.) GOWT1 cells a lot of 0s are in the image
                if np.min(img_crop[:100, :100, ...]) == 0:
                    if np.sum(gaussian_filter(np.squeeze(img_crop), sigma=1) == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):
                        continue
                else:
                    continue
            if np.max(cell_dist_crop) < 0.8:
                continue

            # Remove only partially visible cells in mask for better comparison with tra_gt
            props_crop, n_part = regionprops(mask_crop), 0
            for cell in props_crop:
                if mode == 'GT' and cell.area <= 0.1 * td_settings['min_area'] and td_settings['scale'] == 1:  # needed since tra_gt seeds are smaller
                    n_part += 1
            if (len(mask_ids) - n_part) >= len(tr_ids):  # A: all cells annotated
                crop_quality = 'A'
            elif (len(mask_ids) - n_part) >= 0.8 * len(tr_ids):  # >= 80% of the cells annotated
                crop_quality = 'B'
            else:
                continue

            # Save only needed crops for kit-sch-ge split
            if td_settings['used_crops']:
                if isinstance(slice_idx, int):
                    if not ([subset, frame, '{:02d}'.format(slice_idx), '{:02d}'.format(y), '{:02d}'.format(x), 'train'] in td_settings['used_crops']) \
                            and not ([subset, frame, '{:02d}'.format(slice_idx), '{:02d}'.format(y), '{:02d}'.format(x), 'val'] in td_settings['used_crops']):
                        continue
                else:
                    if not ([subset, frame, '{:02d}'.format(y), '{:02d}'.format(x), 'train'] in td_settings['used_crops']) \
                            and not ([subset, frame, '{:02d}'.format(y), '{:02d}'.format(x), 'val'] in td_settings['used_crops']):
                        continue

            # Save the images
            tiff.imsave(str(path / crop_quality / 'img_{}'.format(crop_name)), img_crop)
            tiff.imsave(str(path / crop_quality / 'mask_{}'.format(crop_name)), mask_crop)
            tiff.imsave(str(path / crop_quality / 'dist_cell_{}'.format(crop_name)), cell_dist_crop)
            tiff.imsave(str(path / crop_quality / 'dist_neighbor_{}'.format(crop_name)), neighbor_dist_crop)

            # Increase crop counter
            crop_idx += 1

            if mode == 'ST' and crop_idx > td_settings['st_limit']:
                # In the first release of this code, for 3D ST data all possible crops have been created and selected
                # afterwards. So, there is a small discrepancy but the old version needed too much time.
                return crop_idx

    return crop_idx


def get_crop(x, y, crop_size, *imgs):
    """ Get crop from image.

    :param x: Grid position (x-dim).
        :type x: int
    :param y: Grid position (y-dim).
        :type y: int
    :param crop_size: size of the (square) crop
        :type crop_size: int
    :param imgs: Images to crop.
        :type imgs:
    :return: img crop.
    """

    imgs_crop = []

    for img in imgs:
        img_crop = img[y * crop_size:(y + 1) * crop_size, x * crop_size:(x + 1) * crop_size, :]
        imgs_crop.append(img_crop)

    return imgs_crop


def get_annotated_gt_frames(path_train_set):
    """ Get GT frames (so that these frames are not used for STs in GT+ST setting).

    :param path_train_set: path to Cell Tracking Challenge training data sets.
        :type path_train_set: pathlib Path object
    :return: List of available GT frames
    """

    seg_gt_ids_01 = sorted((path_train_set / '01_GT' / 'SEG').glob('*.tif'))
    seg_gt_ids_02 = sorted((path_train_set / '02_GT' / 'SEG').glob('*.tif'))

    annotated_gt_frames = []
    for seg_gt_id in seg_gt_ids_01:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            annotated_gt_frames.append("01_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            annotated_gt_frames.append("01_{}".format(seg_gt_id.stem.split('man_seg')[-1]))
    for seg_gt_id in seg_gt_ids_02:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            annotated_gt_frames.append("02_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            annotated_gt_frames.append("02_{}".format(seg_gt_id.stem.split('man_seg')[-1]))

    return annotated_gt_frames


def get_file(path):
    """ Load json file.

    :param path: Path to the json file to load.
        :type path: pathlib Path object
    """
    with open(path) as f:
        file = json.load(f)
    return file


def get_kernel(cell_type):
    """ Get kernel for morphological closing and opening operation.

    :param cell_type: Cell type/dataset for which the kernel is needed.
        :type cell_type: str
    :return: kernel for closing, kernel for opening
    """

    # Larger cells need larger kernels (could be coupled with mean major axis length in future)
    if cell_type in ['Fluo-C3DH-H157', 'Fluo-N3DH-CHO']:
        kernel_closing = np.ones((20, 20))
        kernel_opening = np.ones((20, 20))
    elif cell_type == 'Fluo-C3DL-MDA231':
        kernel_closing = np.ones((3, 3))
        kernel_opening = np.ones((3, 3))
    elif cell_type == 'Fluo-N3DH-CE':
        kernel_closing = np.ones((15, 15))
        kernel_opening = np.ones((15, 15))
    else:
        kernel_closing = np.ones((10, 10))
        kernel_opening = np.ones((10, 10))

    return kernel_closing, kernel_opening


def get_mask_ids(path_data, ct, mode, split, st_limit):
    """ Get ids of the masks of a specific cell type/dataset.

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Pathlib Path object.
    :param ct: cell type/dataset.
        :type ct: str
    :param mode: 'GT' use gold truth, 'ST' use silver truth.
        :type mode: str
    :param split: Use a single ('01'/'02) or both subsets ('01+02') to select the training data from. 'kit-sch-ge'
            reproduces the training data of the Cell Tracking Challenge team KIT-Sch-GE.
        :type split: str
    :param st_limit: Maximum amount of ST crops to create.
        :type st_limit: int
    :return: mask ids, increment for selecting slices.
    """

    # Get mask ids
    mask_ids_01, mask_ids_02 = [], []
    if '01' in split or split == 'kit-sch-ge':
        mask_ids_01 = sorted((path_data / ct / '01_{}'.format(mode) / 'SEG').glob('*.tif'))
    if '02' in split or split == 'kit-sch-ge':
        mask_ids_02 = sorted((path_data / ct / '02_{}'.format(mode) / 'SEG').glob('*.tif'))
    mask_ids = mask_ids_01 + mask_ids_02

    # Go through each slice for 3D annotations if not stated otherwise later
    slice_increment = 1

    # Reduce amount of STs depending on the available amount of STs (mainly to reduce computation time)
    if mode == 'ST' and split != 'kit-sch-ge':
        
        if len(mask_ids) > st_limit // 2:  # Reduce amount of STs and do not just use all
            if '3D' in ct:  # Assumption: 3D STs are for late frames maybe not that good --> better use first frames
                mask_ids = mask_ids_01[:int(st_limit // 2.5)] + mask_ids_02[:int(st_limit // 2.5)]
            else:
                if len(mask_ids) > 1000:  # High temporal resolution or many cell divisions --> use more late frames
                    mask_ids = mask_ids_01[:1000:10] + mask_ids_01[1000::5] + mask_ids_02[:1000:10] + mask_ids_02[1000::5]
                else:  # Use only half of the frames but increase distance between selected frames
                    mask_ids = mask_ids[::2]

        if '3D' in ct:  # Reduce further amount of STs for 3D data (each slice is a training sample ...)
            if len(tiff.imread(str(mask_ids[0]))) > 40:
                mask_ids = mask_ids[::2]
                slice_increment = 4
            elif len(tiff.imread(str(mask_ids[0]))) > 30:
                mask_ids = mask_ids[::2]
                slice_increment = 2
            else:
                slice_increment = 1

    if mode == 'ST' and split == 'kit-sch-ge':
        # No need to reduce data (done later --> only used frames/slices are calculated) but the slice increments
        # accelerate the training data generation.
        if '3D' in ct:  # Reduce further amount of STs for 3D data (each slice is a training sample ...)
            if len(tiff.imread(str(mask_ids[0]))) > 40:
                slice_increment = 4
            elif len(tiff.imread(str(mask_ids[0]))) > 30:
                slice_increment = 2
            else:
                slice_increment = 1
        
    # Shuffle list
    if not split == 'kit-sch-ge':
        shuffle(mask_ids)
                
    return mask_ids, slice_increment


def get_td_settings(mask_id_list, crop_size):
    """ Get settings for the training data generation.

    :param mask_id_list: List of all segmentation GT ids (list of pathlib Path objects).
        :type mask_id_list: list
    :return: dict with keys 'search_radius', 'min_area', 'max_mal', 'scale', 'crop_size'.
    """

    # Load all GT and get cell parameters to adjust parameters for the distance transform calculation
    diameters, major_axes, areas = [], [], []
    for mask_id in mask_id_list:
        mask = tiff.imread(str(mask_id))
        if len(mask.shape) == 3:
            for i in range(len(mask)):
                props = regionprops(mask[i])
                for cell in props:  # works not as intended for 3D GTs
                    major_axes.append(cell.major_axis_length)
                    diameters.append(cell.equivalent_diameter)
                    areas.append(cell.area)
        else:
            props = regionprops(mask)
            for cell in props:
                major_axes.append(cell.major_axis_length)
                diameters.append(cell.equivalent_diameter)
                areas.append(cell.area)

    # Get maximum and minimum diameter and major axis length and set search radius for distance transform
    max_diameter, min_diameter = int(np.ceil(np.max(np.array(diameters)))), int(np.ceil(np.min(np.array(diameters))))
    mean_diameter, std_diameter = int(np.ceil(np.mean(np.array(diameters)))), int(np.std(np.array(diameters)))
    max_mal = int(np.ceil(np.max(np.array(major_axes))))
    min_area = int(0.95 * np.floor(np.min(np.array(areas))))
    search_radius = mean_diameter + std_diameter

    # Some simple heuristics for large cells. If enough data are available scale=1 should work in most cases
    if max_diameter > 200 and min_diameter > 35:
        if max_mal > 2 * max_diameter:  # very longish and long cells not made for neighbor distance
            scale = 0.5
            search_radius = min_diameter + 0.5 * std_diameter
        elif max_diameter > 300 and min_diameter > 60:
            scale = 0.5
        elif max_diameter > 250 and min_diameter > 50:
            scale = 0.6
        else:
            scale = 0.7
        min_area = (scale ** 2) * min_area
        max_mal = int(np.ceil(scale * max_mal))
        search_radius = int(np.ceil(scale * search_radius))

    else:
        scale = 1

    return {'search_radius': search_radius,
            'min_area': min_area,
            'max_mal': max_mal,
            'scale': scale,
            'crop_size': crop_size}


def get_train_val_split(img_idx_list, b_img_idx_list):
    """ Split generated training data crops into training and validation set.
    :param img_idx_list: List of image indices/paths (list of Pathlib Path objects).
        :type img_idx_list: list
    :param b_img_idx_list: List of image indices/paths which were classified as 'B' (list of Pathlib Path objects).
        :type b_img_idx_list: list
    :return: dict with ids for training and ids for validation.
    """

    img_ids_stem = []
    for idx in img_idx_list:
        img_ids_stem.append(idx.stem.split('img_')[-1])
    # Random 80%/20% split
    shuffle(img_ids_stem)
    # train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
    train_ids = img_ids_stem
    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]
    # Add "B" quality only to train
    for idx in b_img_idx_list:
        train_ids.append(idx.stem.split('img_')[-1])
    # Train/val split
    train_val_ids = {'train': train_ids, 'val': val_ids}

    return train_val_ids


def get_used_crops(train_val_ids, mode='GT'):
    """ Get frames used in given training/validation split.

    :param train_val_ids: Training/validation split ids.
        :type train_val_ids: dict
    :param mode: 'GT' use gold truth, 'ST' use silver truth, 'GT+ST' use mixture of gold and silver truth.
        :type mode: str
    :return: used crops [subset, frame, (slice for 3D data), y grid position, x grid position].
    """

    used_crops = []

    if mode == 'GT+ST':
        # Only ST ids are saved since the GTs are just copied from the corresponding ground truth set
        sets = ['train_st', 'val_st']  # bad selection of train/val key names ...
    else:
        sets = ['train', 'val']
    for split_mode in sets:
        for idx in train_val_ids[split_mode]:
            if '2D' in idx:
                used_crops.append([idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2], idx.split('_')[-1],
                                   split_mode])
            else:
                if idx.split('_')[-5] in ['GT', 'ST', "GT+ST"]:  # only frame annotated --> slice info has not been saved
                    used_crops.append([idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2], idx.split('_')[-1],
                                       split_mode])
                else:
                    used_crops.append([idx.split('_')[-5], idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2],
                                       idx.split('_')[-1], split_mode])

    return used_crops


def make_train_dirs(path):
    """ Make directories to save the created training data into.

    :param path: Path to the created training data sets.
        :type path: pathlib Path object.
    :return: None
    """

    Path.mkdir(path / 'A', parents=True, exist_ok=True)  # for high quality crops
    Path.mkdir(path / 'B', exist_ok=True)  # for good quality crops
    Path.mkdir(path / 'train', exist_ok=True)
    Path.mkdir(path / 'val', exist_ok=True)

    return None


def remove_st_with_gt_annotation(st_ids, annotated_gt_frames):
    """ Remove ST crops which are taken from frames which have also a GT available.

    :param st_ids: List of pathlib Path objects.
        :type st_ids: list
    :param annotated_gt_frames: Annotated GT frames.
        :type annotated_gt_frames: list
    return: None
    """

    # gt_frames: "subset_frame"
    for st_id in st_ids:
        frame = "{}_{}".format(st_id.stem.split('ST_')[-1].split('_')[0], st_id.stem.split('ST_')[-1].split('_')[1])
        if frame in annotated_gt_frames:
            files_to_remove = list(st_id.parent.glob("*{}".format(st_id.name.split('img')[-1])))
            for idx in files_to_remove:
                os.remove(idx)
    return None


def write_file(file, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(file, f, ensure_ascii=False, indent=2)
    return


def create_ctc_training_sets(path_data, mode, cell_type_list, split='01+02', crop_size=320, st_limit=280,
                             n_max_train_gt_st=150, n_max_val_gt_st=30):
    """ Create training sets for Cell Tracking Challenge data.

    In the new version of this code, 2 Fluo-C3DL-MDA231 crops and 1 Fluo-C3DH-H157 crop differ slightly from the
    original kit-sch-ge training sets. This should not make any difference for the model training: in the
    Fluo-C3DH-H157 case only one pixel differs in the segmentation mask, and in the other case just the next slice is
    taken instead of the original slice.

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Pathlib Path object.
    :param mode: 'GT' gold truth, 'ST' silver truth, 'GT+ST': mixture of gold truth and silver truth.
        :type mode: str
    :param cell_type_list: List of cell types to include in the training data set. If more than 1 cell type a unique
            name of the training is built (hash). Special case: cell_type_list = ['all'] (see code below).
        :type cell_type_list: list
    :param split: Use a single ('01'/'02) or both subsets ('01+02') to select the training data from. 'kit-sch-ge'
            reproduces the training data of the Cell Tracking Challenge team KIT-Sch-GE.
        :type split: str
    :param crop_size: Size of the generated crops (square).
        :type crop_size: int
    :param st_limit: Maximum amount of ST crops to create (reduces computation time.
        :type st_limit: int
    :param n_max_train_gt_st: Maximum number of gold truths per cell type in the training sets of training datasets
            consisting of multiple cell types.
        :type n_max_train_gt_st: int
    :param n_max_val_gt_st: Maximum number of gold truths per cell type in the validation sets of training datasets
            consisting of multiple cell types.
        :type n_max_val_gt_st: int
    :return: None
    """

    if split == 'kit-sch-ge':
        st_limit = 280  # needed for reproducibility (smaller values will just not work to reproduce that split)

    # Check if multiple cell types are selected
    if len(cell_type_list) > 1:
        trainset_name = hashlib.sha1(str(cell_type_list).encode("UTF-8")).hexdigest()[:10]
        print('Multiple cell types, dataset name: {}'.format(trainset_name))
    elif cell_type_list[0] == 'all':
        trainset_name = 'all'
        # Use cell types included in the primary track
        cell_type_list = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-MSC", "Fluo-C3DH-A549",
                          "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "Fluo-N3DH-CE",
                          "Fluo-N3DH-CHO", "PhC-C2DH-U373", "PhC-C2DL-PSC"]
    else:
        trainset_name = cell_type_list[0]

    # Create needed training data sets (for multiple selected cell types the data sets are copied together later)
    # For mode == 'GT+ST', the required GT and ST data sets need to be generated first 
    for ct in cell_type_list:

        # Check if data set already exists
        path_trainset = path_data / "{}_{}_{}".format(ct, mode, split)
        if len(list((path_trainset / 'train').glob('*.tif'))) > 0:
            print('   ... training set {} already exists ...'.format(path_trainset.stem))
            continue
        print('   ... create {} training set ...'.format(path_trainset.stem))
        make_train_dirs(path=path_trainset)

        # Load split if original 'kit-sch-ge' training sets should be reproduced
        if split == 'kit-sch-ge':
            train_val_ids = get_file(path=Path(__file__).parent/'splits'/'ids_{}_{}.json'.format(ct, mode))
            used_crops = get_used_crops(train_val_ids, mode)
        else:
            used_crops = []

        if mode == 'GT+ST':  # simply copy existing datasets together
            # Copy GT train/val to GT+ST train/val and get number of GT crops (train/val)
            copy_train_set(path_data / "{}_GT_{}".format(ct, split), path_trainset, mode='GT')
            num_gt_train = len(list((path_trainset / 'train').glob('img*.tif')))
            num_gt_val = len(list((path_trainset / 'val').glob('img*.tif')))
            # Copy ST temporarily to GT+ST folder and get number of STs to add (idea: use more GT than ST)
            copy_train_set(path_data / "{}_ST_{}".format(ct, split), path_trainset, mode='ST')
            num_add_st_train = np.maximum(int(0.33 * num_gt_train), 75 - num_gt_train)
            num_add_st_val = np.maximum(int(0.25 * num_gt_val), 15 - num_gt_val)
            if get_file(path_data / "{}_GT_{}".format(ct, split) / 'info.json')['scale'] != get_file(path_data / "{}_ST_{}".format(ct, split) / 'info.json')['scale']:
                num_add_st_train, num_add_st_val = 1e3, 1e3  # just use all ST due to different scaling
            # Go through ST crops and remove crop if frame has a corresponding GT annotation
            st_train_ids = sorted((path_trainset / 'train_st').glob('img*.tif'))
            st_val_ids = sorted((path_trainset / 'val_st').glob('img*.tif'))
            annotated_gt_frames = get_annotated_gt_frames(path_data / ct)
            remove_st_with_gt_annotation(st_train_ids + st_val_ids, annotated_gt_frames)
            # Get ids of usable crops in ST_train/ST_val
            st_train_ids = list((path_trainset / 'train_st').glob('img*.tif'))
            st_val_ids = list((path_trainset / 'val_st').glob('img*.tif'))
            shuffle(st_train_ids), shuffle(st_val_ids)
            # Set counters for new splits
            counter, counter_val = 0, 0
            # Go through ST train files
            for st_train_id in st_train_ids:
                if split == 'kit-sch-ge':
                    if not (st_train_id.stem.split('img_')[-1] in train_val_ids['train_st']):
                        continue
                else:
                    if counter >= num_add_st_train:
                        continue
                # Copy img, distance labels and mask
                copy_train_data(path_trainset / 'train_st', path_trainset / 'train', st_train_id.stem.split('img_')[-1])
                counter += 1
            # Go through ST val files
            for st_val_id in st_val_ids:
                if split == 'kit-sch-ge':
                    if not (st_val_id.stem.split('img_')[-1] in train_val_ids['val_st']):
                        continue
                else:
                    if counter_val >= num_add_st_val:
                        continue
                # Copy img, distance labels and mask
                copy_train_data(path_trainset / 'val_st', path_trainset / 'val', st_val_id.stem.split('img_')[-1])
                counter_val += 1
            td_settings = {'scale': 1, 'cell_type': ct}  # For simplicity: just scale 1 for all cell types
            write_file(td_settings, path_trainset / 'info.json')
            # Remove temporary directories
            shutil.rmtree(str(path_trainset / 'train_st')), shutil.rmtree(str(path_trainset / 'val_st'))
            shutil.rmtree(str(path_trainset / 'A')), shutil.rmtree(str(path_trainset / 'B'))
            continue

        # Get ids of segmentation ground truth masks (GTs may not be fully annotated and STs may be erroneous)
        mask_ids, slice_increment = get_mask_ids(path_data=path_data, ct=ct, mode=mode, split=split, st_limit=st_limit)

        # Get settings for distance map creation
        td_settings = get_td_settings(mask_id_list=mask_ids, crop_size=crop_size)
        td_settings['used_crops'],  td_settings['st_limit'], td_settings['cell_type'] = used_crops, st_limit, ct

        # Iterate through files and load images and masks (and TRA GT for GT mode)
        running_index = 0
        for mask_id in mask_ids:

            if mode == 'ST' and split != 'kit-sch-ge' and running_index > st_limit:
                continue

            # Load images and masks (get slice and frame first)
            if len(mask_id.stem.split('_')) > 2:  # only slice annotated
                # frame = mask_id.stem.split('_')[-1]
                frame = mask_id.stem.split('_')[2]
                slice_idx = int(mask_id.stem.split('_')[3])
            else:
                frame = mask_id.stem.split('man_seg')[-1]

            # Check if frame is needed to reproduce the kit-sch-ge training sets
            if used_crops and not any(e[1] == frame for e in used_crops):
                continue

            # Load image and mask and get subset from which they are
            mask = tiff.imread(str(mask_id))
            subset = mask_id.parents[1].stem.split('_')[0]
            img = tiff.imread(str(mask_id.parents[2] / subset / "t{}.tif".format(frame)))

            # TRA GT (fully annotated, no region information) to detect fully annotated mask GTs later
            if 'GT' in mode:
                # tra_gt = tiff.imread(str(mask_id.parents[1] / 'TRA' / "man_track{}.tif".format(frame)))
                tra_gt = np.copy(mask)
            else:  # Do not use TRA GT to detect high quality STs (to be able to compare ST and GT results)
                tra_gt = np.copy(mask)

            # FOI correction
            img, mask, tra_gt = foi_correction_train(ct, mode, img, mask, tra_gt)

            # Downsampling
            if td_settings['scale'] != 1:
                img = downscale(img=img, scale=td_settings['scale'], order=2)
                mask = downscale(img=mask, scale=td_settings['scale'], order=0, aa=False)
                tra_gt = downscale(img=tra_gt, scale=td_settings['scale'], order=0, aa=False)

            # Normalization: min-max normalize image to [0, 65535]
            img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
            img = np.clip(img, 0, 65535).astype(np.uint16)

            # Calculate distance transforms, crop and classify crops into 'A' (fully annotated) and 'B' (>80% annotated)
            if len(mask.shape) == 3:  # 3D annotation
                if mode == 'ST':  # Select slices which contain cells first (just looking at masks not sufficient)
                    img_mean, img_std = np.mean(img), np.std(img)
                    for i in range(len(img)):
                        if i % slice_increment == 0:
                            if slice_increment > 1:  # high axial resolution --> + 0.1 * img_std
                                if np.mean(img[i]) < img_mean + 0.1 * img_std or np.sum(mask[i] == 0) < 0.02 * img.shape[1] * img.shape[2]:
                                    continue
                            else:  # low axial resolution --> -0.1 * img_std (just heuristics which seem to work ...)
                                if np.mean(img[i]) < img_mean - 0.1 * img_std or np.sum(mask[i] > 0) < 0.02 * img.shape[1] * img.shape[2]:
                                    continue

                            # Check if slice is needed to reproduce the kit-sch-ge training sets
                            if used_crops and not any(e[1:3] == [frame, "{:02d}".format(i)] for e in used_crops):
                                continue

                            # Get slices
                            img_slice, mask_slice = img[i], mask[i]
                            # Opening + closing
                            kernel_closing, kernel_opening = get_kernel(cell_type=ct)
                            mask_slice = close_mask(mask_slice, True, kernel_closing, kernel_opening)
                            if ct == 'Fluo-N3DH-CE':
                                for nucleus in regionprops(mask_slice):
                                    if nucleus.bbox_area < 20 * 20:
                                        mask_slice[mask_slice == nucleus.label] = 0

                            # No running index --> create all crops and select later
                            running_index = generate_data(img=img_slice, mask=mask_slice, tra_gt=mask_slice,
                                                          td_settings=td_settings, cell_type=ct, mode=mode,
                                                          subset=subset, frame=frame, path=path_trainset,
                                                          slice_idx=i, crop_idx=running_index)

                else:
                    for i in range(len(mask)):
                        img_slice, mask_slice = img[i].copy(), mask[i].copy()
                        if np.max(mask_slice) == 0:  # empty frame
                            continue
                        mask_slice = close_mask(mask=mask_slice, kernel_closing=np.ones((5, 5)))
                        tr_gt_slice = mask_slice.copy()  # assumption: in 3D GT annotations all cells are annotated
                        _ = generate_data(img=img_slice, mask=mask_slice, tra_gt=tr_gt_slice, td_settings=td_settings,
                                          cell_type=ct,  mode=mode, subset=subset, frame=frame, path=path_trainset,
                                          slice_idx=i)
            else:
                if '3D' in ct:  # 3D data but 2D annotation (only for GT)
                    img = img[slice_idx]
                    # Needed seed could be outside the slice --> maximum intensity projection
                    slice_min, slice_max = np.maximum(slice_idx - 2, 0), np.minimum(slice_idx + 2, len(img) - 1)
                    tra_gt = np.max(tra_gt[slice_min:slice_max], axis=0)  # best bring seed size to min_area ...
                    mask = close_mask(mask=mask, kernel_closing=np.ones((5, 5)))

                if mode == 'ST':
                    if ct == 'DIC-C2DH-HeLa':
                        mask = close_mask(mask=mask, apply_opening=True)
                    running_index = generate_data(img=img, mask=mask, tra_gt=mask, td_settings=td_settings,
                                                  cell_type=ct, mode=mode, subset=subset, frame=frame,
                                                  path=path_trainset, crop_idx=running_index)
                else:
                    _ = generate_data(img=img, mask=mask, tra_gt=tra_gt, td_settings=td_settings, cell_type=ct,
                                      mode=mode, subset=subset, frame=frame, path=path_trainset)

        td_settings.pop('used_crops')
        if mode == 'GT':
            td_settings.pop('st_limit')
        write_file(td_settings, path_trainset / 'info.json')

        # Create train/val split
        img_ids, b_img_ids = sorted((path_trainset / 'A').glob('img*.tif')), []
        if mode == 'GT' and len(img_ids) <= 30:  # Use also "B" quality images when too few "A" quality images are available
            b_img_ids = sorted((path_trainset / 'B').glob('img*.tif'))
        if not split == 'kit-sch-ge':
            train_val_ids = get_train_val_split(img_ids, b_img_ids)

        # Copy images to train/val
        for train_mode in ['train', 'val']:
            for idx in train_val_ids[train_mode]:
                if (path_trainset / "A" / ("img_{}.tif".format(idx))).exists():
                    source_path = path_trainset / "A"
                else:
                    source_path = path_trainset / "B"
                copy_train_data(source_path, path_trainset / train_mode, idx)

    if len(cell_type_list) > 1:  # Copy train and val sets together but avoid to high imbalance of cell types

        if trainset_name == 'all':
            if mode == 'GT+ST':
                path_trainset = path_data / "allGT+allST_{}".format(split)
            else:
                path_trainset = path_data / "{}{}_{}".format(trainset_name, mode, split)
        else:
            path_trainset = path_data / "{}_{}_{}".format(trainset_name, mode, split)
        if len(list((path_trainset / 'train').glob('*.tif'))) > 0:
            print('   ... training set {} already exists ...'.format(path_trainset.stem))
            return
        print('   ... create {} training set ...'.format(path_trainset.stem))
        make_train_dirs(path=path_trainset)

        # Load split if original 'kit-sch-ge' training sets should be reproduced
        train_val_ids = {}
        if split == 'kit-sch-ge':
            if mode == 'GT+ST':
                train_val_ids = get_file(path=Path(__file__).parent/'splits'/'ids_allGT+allST.json')
            else:
                train_val_ids = get_file(path=Path(__file__).parent/'splits'/'ids_{}{}.json'.format(trainset_name, mode))

        cell_counts = {'train': {}, 'val': {}}
        for ct in cell_type_list:

            if 'all' in trainset_name and ct == 'Fluo-C2DL-MSC' and mode == 'GT':  # use no c2dl-msc data since too different to other cells
                cell_counts['train'][ct], cell_counts['val'][ct] = 0, 0
                continue

            img_ids = {'train': sorted((path_data / "{}_{}_{}".format(ct, mode, split) / 'train').glob('img*.tif')),
                       'val': sorted((path_data / "{}_{}_{}".format(ct, mode, split) / 'val').glob('img*.tif'))}

            if mode == 'GT+ST':
                if not train_val_ids:
                    n_max = {'train': n_max_train_gt_st, 'val': n_max_val_gt_st}  # maximum number of each cell type
                    if '3D' in ct or ct == 'Fluo-C2DL-MSC':
                        n_max = {'train': n_max_train_gt_st // 2, 'val': n_max_val_gt_st // 2}
                img_ids = {'train': sorted((path_data / "{}_GT+ST_{}".format(ct, split) / 'train').glob('img*.tif')),
                           'val': sorted((path_data / "{}_GT+ST_{}".format(ct, split) / 'val').glob('img*.tif'))}
                shuffle(img_ids['train']), shuffle(img_ids['val'])
                counter = {'train': 0, 'val': 0}
                for train_mode in ['train', 'val']:
                    for img_id in img_ids[train_mode]:
                        if not train_val_ids:
                            if counter[train_mode] >= n_max[train_mode]:
                                continue
                        else:  # ids available
                            if not (img_id.stem.split('img_')[-1] in train_val_ids[train_mode]):
                                continue
                        counter[train_mode] += 1
                        # Copy img, distance labels and mask
                        copy_train_data(img_id.parent, path_trainset / train_mode, img_id.stem.split('img_')[-1])
                    cell_counts[train_mode][ct] = counter[train_mode]
                continue

            if not train_val_ids:
                p_neighbor, p_no_neighbor = 1, 1  # keep all crops with neighbor information (exceptions below)
                if mode == 'ST':
                    p_neighbor = 0.9  # keep (almost) all crops with neighbor distance information
                    p_no_neighbor = 0.6  # keep more than half of the other crops
                    if ct in ['Fluo-C2DL-MSC', 'DIC-C2DH-HeLa']:
                        p_neighbor = 0.4
                        p_no_neighbor = 0.2
                    if ct in ['Fluo-C3DL-MDA231', 'Fluo-C3DH-H157']:
                        p_neighbor = 0.6
                        p_no_neighbor = 0.4
                elif mode == 'GT':
                    if "3D" in ct:
                        if len(img_ids['train']) + len(img_ids['val']) > 100:
                            p_no_neighbor = 0.15  # use only 15% of images with dist_neighbor = 0
                        elif len(img_ids['train']) + len(img_ids['val']) > 50:
                            p_no_neighbor = 0.75  # use 3/4 of images with dist_neighbor = 0
                    else:
                        if len(img_ids['train']) + len(img_ids['val']) > 150:
                            p_no_neighbor = 0.5
                        elif len(img_ids['train']) + len(img_ids['val']) > 75:
                            p_no_neighbor = 0.75

                for train_mode in ['train', 'val']:
                    counts = 0
                    hlist = []
                    for idx in img_ids[train_mode]:
                        fname = idx.stem.split('img_')[-1]
                        if np.sum(tiff.imread(str(idx.parent / 'dist_neighbor_{}.tif'.format(fname))) > 0) == 0:
                            if random() > p_no_neighbor:
                                continue
                        elif p_neighbor < 1:
                            if random() > p_neighbor:
                                continue
                        hlist.append(fname.split('.tif')[0])
                        source_path = path_data / "{}_{}_{}".format(ct, mode, split) / train_mode
                        copy_train_data(source_path, path_trainset / train_mode, fname)
                        counts += 1
                    cell_counts[train_mode][ct] = counts
            else:
                for train_mode in ['train', 'val']:
                    counts = 0
                    for idx in img_ids[train_mode]:
                        fname = idx.stem.split('img_')[-1]
                        if ct in train_val_ids[train_mode].keys():
                            if fname.split('.tif')[0] in train_val_ids[train_mode][ct]:
                                source_path = path_data / "{}_{}_{}".format(ct, mode, split) / train_mode
                                copy_train_data(source_path, path_trainset / train_mode, fname)
                                counts += 1
                    cell_counts[train_mode][ct] = counts
        cell_counts['scale'] = 1
        write_file(cell_counts, path_trainset / 'info.json')
