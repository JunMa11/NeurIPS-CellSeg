import argparse
import numpy as np
import random
import torch
import warnings

from copy import deepcopy
from pathlib import Path

from segmentation.inference.inference import inference_2d_ctc, inference_3d_ctc

warnings.filterwarnings("ignore", category=UserWarning)


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='KIT-Sch-GE 2021 Cell Segmentation - Inference')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--cell_type', '-ct', nargs='+', required=True, help='Cell type(s) to predict')
    parser.add_argument('--fuse_z_seeds', '-fzs', default=False, action='store_true', help='Fuse seeds in axial direction')
    parser.add_argument('--model', '-m', required=True, type=str, help='Name of the model to use')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int, help='Cell amount threshold to apply splitting post-processing (3D)')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    parser.add_argument('--th_cell', '-tc', default=0.07, type=float, help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, type=float, help='Threshold for seeds')
    args = parser.parse_args()

    # Paths
    path_data = Path(__file__).parent / 'challenge_data'
    path_models = Path(__file__).parent / 'models'

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    # Check if dataset consists in challenge_data folder
    for ct in args.cell_type:
        if not (path_data / ct).exists():
            print('No data for "{}" found in {}'.format(ct, path_data))
            return

    for ct in args.cell_type:

        # Load model
        model = path_models / args.model
        model = model.parent / "{}.pth".format(model.stem)

        subsets = [args.subset]
        if args.subset in ['kit-sch-ge', '01+02']:
            subsets = ['01', '02']

        for subset in subsets:
            # Directory for results
            path_seg_results = path_data / ct / "{}_RES_{}".format(subset, model.stem)
            path_seg_results.mkdir(exist_ok=True)
            print('Inference using {} on {}_{}: th_seed: {}, th_cell: {}, scale: {}'.format(model.stem,
                                                                                            ct,
                                                                                            subset,
                                                                                            args.th_seed,
                                                                                            args.th_cell,
                                                                                            args.scale))
            # Check if results already exist
            if len(sorted(path_seg_results.glob('*.tif'))) > 0:
                print('   Segmentation results already exist (delete for new calculation).')
                continue

            inference_args = deepcopy(args)
            inference_args.cell_type = ct

            if '2D' in ct:
                inference_2d_ctc(model=model,
                                 data_path=path_data / ct / subset,
                                 result_path=path_seg_results,
                                 device=device,
                                 batchsize=args.batch_size,
                                 args=inference_args,
                                 num_gpus=num_gpus)
            else:
                inference_3d_ctc(model=model,
                                 data_path=path_data / ct / subset,
                                 result_path=path_seg_results,
                                 device=device,
                                 batchsize=args.batch_size,
                                 args=inference_args,
                                 num_gpus=num_gpus)


if __name__ == "__main__":

    main()
