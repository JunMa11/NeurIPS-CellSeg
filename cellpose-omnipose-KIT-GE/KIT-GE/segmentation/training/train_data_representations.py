import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, grey_closing, generate_binary_structure
from skimage import measure
from skimage.morphology import disk
from segmentation.utils.utils import get_nucleus_ids


def bottom_hat_closing(label):
    """ Bottom-hat-transform based grayscale closing.

    :param label: Intensity coded label image.
        :type label:
    :return: closed label (only closed regions, all 1), closed label (only closed regions, 0.8-1.0)
    """

    label_bin = np.zeros_like(label, dtype=bool)

    # Apply closing to each nucleus to avoid artifacts
    nucleus_ids = get_nucleus_ids(label)
    for nucleus_id in nucleus_ids:
        nucleus = (label == nucleus_id)
        nucleus = ndimage.binary_closing(nucleus, disk(3))
        label_bin[nucleus] = True

    # Bottom-hat-transform
    label_bottom_hat = ndimage.binary_closing(label_bin, disk(3)) ^ label_bin
    label_closed = (~label_bin) & label_bottom_hat

    # Integrate gaps better into the neighbor distances
    label_closed = measure.label(label_closed.astype(np.uint8))
    props = measure.regionprops(label_closed)
    label_closed_corr = (label_closed > 0).astype(np.float32)
    for i in range(len(props)):
        if props[i].minor_axis_length >= 3:
            single_gap = label_closed == props[i].label
            single_gap_border = single_gap ^ ndimage.binary_erosion(single_gap, generate_binary_structure(2, 1))
            label_closed_corr[single_gap] = 1
            label_closed_corr[single_gap_border] = 0.8  # gets scaled later to 0.84

    return label_closed, label_closed_corr


def border_label_2d(label):
    """ Border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Border label image.
    """

    label_bin = label > 0
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    # Pre-allocation
    boundary = np.zeros(shape=label.shape, dtype=np.bool)

    nucleus_ids = get_nucleus_ids(label)

    for nucleus_id in nucleus_ids:
        nucleus = (label == nucleus_id)
        nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
        boundary += nucleus_boundary

    border = boundary ^ (ndimage.binary_dilation(label_bin, kernel) ^ label_bin)
    label_border = np.maximum(label_bin, 2 * border)

    return label_border


def distance_label_2d(label, cell_radius, neighbor_radius):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param cell_radius: Defines the area to compute the distance transform for the cell distances.
        :type cell_radius: int
    :param neighbor_radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type neighbor_radius: int
    :return: Cell distance label image, neighbor distance label image.
    """

    # Preallocation
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    # Get Borders in-between touching cells
    label_border = (border_label_2d(label) == 2)

    # Find centroids, crop image, calculate distance transforms
    props = measure.regionprops(label)
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - cell_radius, 0)):int(min(centroid[0] + cell_radius, label.shape[0])),
                       int(max(centroid[1] - cell_radius, 0)):int(min(centroid[1] + cell_radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        max_dist = np.max(nucleus_crop_dist)
        if max_dist > 0:
            nucleus_crop_dist = nucleus_crop_dist / max_dist
        else:
            continue
        label_dist[
        int(max(centroid[0] - cell_radius, 0)):int(min(centroid[0] + cell_radius, label.shape[0])),
        int(max(centroid[1] - cell_radius, 0)):int(min(centroid[1] + cell_radius, label.shape[1]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1]))
                                ])

        # No need to calculate neighbor distances if no neighbor is in the crop
        if len(get_nucleus_ids(nucleus_neighbor_crop)) <= 1:
            continue

        # Convert background to nucleus id
        nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
        nucleus_neighbor_crop = nucleus_neighbor_crop > 0
        nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
        nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
        if np.max(nucleus_neighbor_crop_dist) > 0:
            denominator = np.minimum(max_dist + 3,  # larger than max_dist since scaled later on (improves small cells)
                                     np.max(nucleus_neighbor_crop_dist))
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / denominator
            nucleus_neighbor_crop_dist = np.clip(nucleus_neighbor_crop_dist, 0, 1)
        else:
            nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1]))
        ] += nucleus_neighbor_crop_dist

    # Add neighbor distances in-between close but not touching cells with bottom-hat transform / fill gaps
    label_closed, label_closed_corr = bottom_hat_closing(label=label)
    props = measure.regionprops(label_closed)
    # Remove artifacts in the gap class
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    for obj_props in props:
        obj = (label_closed == obj_props.label)
        # There should be no high grayscale values around artifacts
        obj_boundary = ndimage.binary_dilation(obj, kernel) ^ obj
        if obj_props.area <= 20:
            th = 5
        elif obj_props.area <= 30:
            th = 8
        elif obj_props.area <= 50:
            th = 10
        else:
            th = 20
        if np.sum(obj_boundary * label_dist_neighbor) < th:  # Complete in background
            label_closed_corr[obj] = 0

    # label_dist_neighbor = np.maximum(label_dist_neighbor, label_gap.astype(label_dist_neighbor.dtype))
    label_dist_neighbor = np.maximum(label_dist_neighbor, label_closed_corr.astype(label_dist_neighbor.dtype))
    label_dist_neighbor = np.maximum(label_dist_neighbor, label_border.astype(label_dist_neighbor.dtype))

    # Scale neighbor distances
    label_dist_neighbor = 1 / np.sqrt(0.65 + 0.5 * np.exp(-11 * (label_dist_neighbor - 0.75))) - 0.19
    label_dist_neighbor = np.clip(label_dist_neighbor, 0, 1)
    label_dist_neighbor = grey_closing(label_dist_neighbor, size=(3, 3))

    return label_dist.astype(np.float32), label_dist_neighbor.astype(np.float32)

