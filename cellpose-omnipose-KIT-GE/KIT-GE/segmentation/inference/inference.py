import gc
import json
import tifffile as tiff
import torch

from multiprocessing import cpu_count
from skimage.measure import regionprops, label
from skimage.transform import resize

from segmentation.inference.ctc_dataset import pre_processing_transforms, CTCDataSet_test
from segmentation.inference.postprocessing import *
from segmentation.utils.unets import build_unet


def inference_2d_ctc(model, data_path, result_path, device, batchsize, args, num_gpus=None):
    """ Inference function for 2D Cell Tracking Challenge data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param data_path: Path to the directory containing the Cell Tracking Challenge data sets.
        :type data_path: pathlib Path object
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param batchsize: Batch size.
        :type batchsize: int
    :param args: Arguments for post-processing.
        :type args:
    :param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        :type num_gpus: int
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    # Build model
    net = build_unet(unet_type=model_settings['architecture'][0],
                     act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=1,
                     ch_out=1,
                     filters=model_settings['architecture'][4])

    # Get number of GPUs to use and load weights
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    torch.set_grad_enabled(False)

    # Get images to predict
    ctc_dataset = CTCDataSet_test(data_dir=data_path,
                             transform=pre_processing_transforms(apply_clahe=args.apply_clahe, scale_factor=args.scale))
    if device.type == "cpu":
        num_workers = 0
    else:
        try:
            num_workers = cpu_count() // 2
        except AttributeError:
            num_workers = 4
    if num_workers <= 2:  # Probably Google Colab --> use 0
        num_workers = 0
    num_workers = np.minimum(num_workers, 16)
    dataloader = torch.utils.data.DataLoader(ctc_dataset, batch_size=batchsize, shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    # Predict images (iterate over images/files)
    for sample in dataloader:
        img_batch, ids_batch, pad_batch, img_size = sample
        img_batch = img_batch.to(device)

        if batchsize > 1:  # all images in a batch have same dimensions and pads
            pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]
            img_size = [img_size[i][0] for i in range(len(img_size))]

        # Prediction
        prediction_border_batch, prediction_cell_batch = net(img_batch)

        # Get rid of pads
        prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
        prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        # Save also some raw predictions (not all since float32 --> needs lot of memory)
        save_ids = [0, len(ctc_dataset) // 8, len(ctc_dataset) // 4, 3 * len(ctc_dataset) // 8, len(ctc_dataset) // 2,
                    5 * len(ctc_dataset), 3 * len(ctc_dataset) // 4, 7 * len(ctc_dataset) // 8, len(ctc_dataset) - 1]

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_border_batch)):

            print('         ... processing {0} ...'.format(ids_batch[h]))


            save_raw_pred = False

            file_id = ids_batch[h] + '_label.tiff'
            prediction_instance, border = distance_postprocessing(border_prediction=prediction_border_batch[h],
                                                                  cell_prediction=prediction_cell_batch[h],
                                                                  args=args)
            if args.scale < 1:
                prediction_instance = resize(prediction_instance,
                                             img_size,
                                             order=0,
                                             preserve_range=True,
                                             anti_aliasing=False).astype(np.uint16)

            prediction_instance = foi_correction(mask=prediction_instance, cell_type=args.cell_type)

            tiff.imsave(str(result_path / (file_id)), prediction_instance, compress=1)
            if save_raw_pred:
                tiff.imsave(str(result_path / ('cell' + file_id)), prediction_cell_batch[h, ..., 0].astype(np.float32), compress=1)
                tiff.imsave(str(result_path / ('raw_border' + file_id)), prediction_border_batch[h, ..., 0].astype(np.float32), compress=1)
                tiff.imsave(str(result_path / ('border' + file_id)), border.astype(np.float32), compress=1)

    if args.artifact_correction:
        # Artifact correction based on the assumption that the cells are dense and artifacts far away
        roi = np.zeros_like(prediction_instance) > 0
        prediction_instance_ids = sorted(result_path.glob('mask*'))
        for prediction_instance_id in prediction_instance_ids:
            roi = roi | (tiff.imread(str(prediction_instance_id)) > 0)
        roi = binary_dilation(roi, np.ones(shape=(20, 20)))
        roi = label(roi)
        props = regionprops(roi)
        # Keep only the largest region
        largest_area, largest_area_id = 0, 0
        for prop in props:
            if prop.area > largest_area:
                largest_area = prop.area
                largest_area_id = prop.label
        roi = (roi == largest_area_id)
        for prediction_instance_id in prediction_instance_ids:
            prediction_instance = tiff.imread(str(prediction_instance_id))
            prediction_instance = prediction_instance * roi
            tiff.imsave(str(prediction_instance_id), prediction_instance.astype(np.uint16), compress=1)

    # Clear memory
    del net
    gc.collect()

    return None
