import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing

from segmentation.utils.utils import get_nucleus_ids


def foi_correction(mask, cell_type):
    """ Field of interest correction for Cell Tracking Challenge data (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    :param mask: Segmented cells.
        :type mask:
    :param cell_type: Cell Type.
        :type cell_type: str
    :return: FOI corrected segmented cells.
    """

    if cell_type in ['DIC-C2DH-HeLa', 'Fluo-C2DL-Huh7', 'Fluo-C2DL-MSC', 'Fluo-C3DH-H157', 'Fluo-N2DH-GOWT1',
                     'Fluo-N3DH-CE', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
        E = 50
    elif cell_type in ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'Fluo-C3DL-MDA231', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']:
        E = 25
    else:
        E = 0

    if len(mask.shape) == 2:
        foi = mask[E:mask.shape[0] - E, E:mask.shape[1] - E]
    else:
        foi = mask[:, E:mask.shape[1] - E, E:mask.shape[2] - E]

    ids_foi = get_nucleus_ids(foi)
    ids_prediction = get_nucleus_ids(mask)
    for id_prediction in ids_prediction:
        if id_prediction not in ids_foi:
            mask[mask == id_prediction] = 0

    return mask


def distance_postprocessing(border_prediction, cell_prediction, args, input_3d=False):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        :type args:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    if input_3d:
        sigma_cell = (0.5, 1.0, 1.0)
    else:
        sigma_cell = 0.5

    apply_splitting = False

    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
    border_prediction = np.clip(border_prediction, 0, 1)

    th_seed = args.th_seed
    th_cell = args.th_cell
    th_local = 0.25

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = np.tan(border_prediction ** 2)
    borders[borders < 0.05] = 0
    borders = np.clip(borders, 0, 1)
    cell_prediction_cleaned = (cell_prediction - borders)
    seeds = cell_prediction_cleaned > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    areas = []
    for i in range(len(props)):
        areas.append(props[i].area)
    if len(areas) > 0:
        min_area = 0.10 * np.mean(np.array(areas))
    else:
        min_area = 0
    min_area = np.maximum(min_area, 8) if input_3d else np.maximum(min_area, 4)

    for i in range(len(props)):
        # if props[i].area <= 4 or (input_3D and props[i].area <= 8):
        if props[i].area <= min_area:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Avoid empty predictions (there needs to be at least one cell)
    while np.max(seeds) == 0 and th_seed > 0.05:
        th_seed -= 0.1
        seeds = cell_prediction_cleaned > th_seed
        seeds = measure.label(seeds, background=0)
        props = measure.regionprops(seeds)
        for i in range(len(props)):
            if props[i].area <= 4 or (input_3d and props[i].area <= 8):
                seeds[seeds == props[i].label] = 0
        seeds = measure.label(seeds, background=0)

    # local splitting since the slice-wise predictions tend to undersegmentation
    if input_3d and np.max(seeds) >= args.n_splitting:

        seeds = ((cell_prediction - 0.5 * borders) > th_local)  # give chance to correct wrong borders
        seeds = measure.label(seeds, background=0)

        # Remove very small seeds
        props = measure.regionprops(seeds)
        for i in range(len(props)):
            if props[i].area <= 16:  # Due to the smaller threshold also bigger seeds need to be removed
                seeds[seeds == props[i].label] = 0
        seeds = measure.label(seeds, background=0)

        prediction = cell_prediction * (seeds > 0)

        seed_list = peak_local_max(prediction, min_distance=6)
        seeds = np.zeros_like(prediction)
        for seed in range(len(seed_list)):
            seeds[seed_list[seed][0], seed_list[seed][1], seed_list[seed][2]] = 1
        seeds = measure.label(seeds)

        apply_splitting = True

    if args.fuse_z_seeds:
        seeds = seeds > 0
        kernel = np.ones(shape=(3, 1, 1))
        seeds = binary_closing(seeds, kernel)
        seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    if args.apply_merging and np.max(prediction_instance) < 255:
        # Get borders between touching cells
        label_bin = prediction_instance > 0
        pred_boundaries = cv2.Canny(prediction_instance.astype(np.uint8), 1, 1) > 0
        pred_borders = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
        pred_borders = pred_boundaries ^ pred_borders
        pred_borders = measure.label(pred_borders)
        for border_id in get_nucleus_ids(pred_borders):
            pred_border = (pred_borders == border_id)
            if np.sum(border_prediction[pred_border]) / np.sum(pred_border) < 0.075:  # very likely splitted due to shape
                # Get ids to merge
                pred_border_dilated = binary_dilation(pred_border, np.ones(shape=(3, 3), dtype=np.uint8))
                merge_ids = get_nucleus_ids(prediction_instance[pred_border_dilated])
                if len(merge_ids) == 2:
                    prediction_instance[prediction_instance == merge_ids[1]] = merge_ids[0]
        prediction_instance = measure.label(prediction_instance)

    # Iterative splitting of cells detected as (probably) merged
    if apply_splitting:
        props = measure.regionprops(prediction_instance)
        volumes, nucleus_ids = [], []
        for i in range(len(props)):
            volumes.append(props[i].area)
            nucleus_ids.append(props[i].label)
        volumes = np.array(volumes)
        for i, nucleus_id in enumerate(nucleus_ids):
            if volumes[i] > np.mean(volumes) + 2/5 * np.mean(volumes):
                nucleus_bin = (prediction_instance == nucleus_id)
                cell_prediction_nucleus = cell_prediction * nucleus_bin
                for th in [0.50, 0.60, 0.75]:
                    new_seeds = measure.label(cell_prediction_nucleus > th)
                    if np.max(new_seeds) > 1:
                        new_cells = watershed(image=-cell_prediction_nucleus, markers=new_seeds, mask=nucleus_bin,
                                              watershed_line=False)
                        new_ids = get_nucleus_ids(new_cells)
                        for new_id in new_ids:
                            prediction_instance[new_cells == new_id] = np.max(prediction_instance) + 1
                        break

    return np.squeeze(prediction_instance.astype(np.uint16)), np.squeeze(borders)


