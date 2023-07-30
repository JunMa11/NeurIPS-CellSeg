import argparse
import json

import numpy as np
import os
import pandas as pd
import random
import torch
import warnings

from pathlib import Path

from segmentation.inference.inference import inference_2d_ctc, inference_3d_ctc
from segmentation.training.create_training_sets import get_file, write_file
from segmentation.utils.metrics import count_det_errors, ctc_metrics
from segmentation.utils import utils

warnings.filterwarnings("ignore", category=UserWarning)


class EvalArgs(object):
    """ Class with post-processing parameters.

    """

    def __init__(self, th_cell, th_seed, n_splitting, apply_clahe, scale, cell_type, save_raw_pred,
                 artifact_correction, apply_merging, fuse_z_seeds):
        """

        :param th_cell: Mask / cell size threshold.
            :type th_cell: float
        :param th_seed: Seed / marker threshold.
            :type th_seed: float
        :param n_splitting: Number of detected cells above which to apply additional splitting (only for 3D).
            :type n_splitting: int
        :param apply_clahe: Apply contrast limited adaptive histogram equalization (CLAHE).
            :type apply_clahe: bool
        :param scale: Scale factor for downsampling.
            :type scale: float
        :param cell_type: Cell type.
            :type cell_type: str
        :param save_raw_pred: Save (some) raw predictions.
            :type save_raw_pred: bool
        :param artifact_correction: Apply artifact correction post-processing.
            :type artifact_correction: bool
        :param fuse_z_seeds: Fuse seeds in z-direction / axial direction.
            :type fuse_z_seeds: bool
        """
        self.th_cell = th_cell
        self.th_seed = th_seed
        self.n_splitting = n_splitting
        self.apply_clahe = apply_clahe
        self.scale = scale
        self.cell_type = cell_type
        self.save_raw_pred = save_raw_pred
        self.artifact_correction = artifact_correction
        self.fuse_z_seeds = fuse_z_seeds
        self.apply_merging = apply_merging


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='KIT-GE (3) / KIT-Sch-GE (2) Cell Segmentation - Evaluation')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--cell_type', '-ct', nargs='+', required=True, help='Cell type(s)')
    parser.add_argument('--fuse_z_seeds', '-fzs', default=False, action='store_true', help='Fuse seeds in axial direction')
    parser.add_argument('--mode', '-m', default='GT', type=str, help='Ground truth type / evaluation mode')
    parser.add_argument('--models', required=True, type=str, help='Models to evaluate (prefix)')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int, help='Cell amount threshold to apply splitting post-processing (3D)')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=0, type=float, help='Scale factor (0: get from trainset info.json')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    parser.add_argument('--th_cell', '-tc', default=0.07, nargs='+', help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, nargs='+', help='Threshold for seeds')
    args = parser.parse_args()

    # Paths
    path_data = Path(__file__).parent / 'training_data'
    path_models = Path(__file__).parent / 'models' / 'all'
    path_best_models = Path(__file__).parent / 'models' / 'best'
    path_ctc_metric = Path(__file__).parent / 'evaluation_software'

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    # Check if dataset consists in training_data folder
    if len(args.cell_type) > 1:
        es = 0
        for cell_type in args.cell_type:
            if not (path_data / cell_type).exists():
                print('No data for cell type "{}" found in {}'.format(cell_type, path_data))
                es = 1
        if es == 1:
            return
        trainset_name = hash(tuple(sorted(args.cell_type)))
    else:
        if not (args.cell_type[0] == 'all') and not (path_data / args.cell_type[0]).exists():
            print('No data for cell type "{}" found in {}'.format(args.cell_type[0], path_data))
            return
        trainset_name = args.cell_type[0]

    # Get cell types / datasets to evaluate
    cell_type_list = args.cell_type
    if args.cell_type[0] == 'all':
        # Use cell types included in the primary track
        cell_type_list = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-MSC", "Fluo-C3DH-A549",
                          "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "Fluo-N3DH-CE",
                          "Fluo-N3DH-CHO", "PhC-C2DH-U373", "PhC-C2DL-PSC"]

    # Check if evaluation metric is available
    if not path_ctc_metric.is_dir():
        raise Exception('No evaluation software found. Run the skript download_data.py')

    # Get models and cell types to evaluate
    models = sorted(path_models.glob("{}*.pth".format(args.models)))
    if len(models) == 0:
        raise Exception('No models to evaluate found.')

    if not isinstance(args.th_seed, list):
        args.th_seed = [args.th_seed]
    if not isinstance(args.th_cell, list):
        args.th_cell = [args.th_cell]

    # Go through model list and evaluate for stated cell_types
    scores = []
    for model in models:
        for ct in cell_type_list:
            train_sets = [args.subset]
            # Get scale from training dataset info if not stated otherwise
            scale_factor = args.scale
            if args.scale == 0:
                scale_factor = get_file(path_data / model.stem.split('_model')[0] / "info.json")['scale']
            # Go through thresholds
            for th_seed in args.th_seed:
                for th_cell in args.th_cell:
                    if args.subset in ['kit-sch-ge', '01+02']:
                        train_sets = ['01', '02']
                    results = {'cell type': ct,
                               'model': model.stem,
                               'th_seed': float(th_seed),
                               'th_cell': float(th_cell),
                               'n_splitting': args.n_splitting if '3D' in ct else False,
                               'apply_clahe': args.apply_clahe,
                               'scale_factor': scale_factor,
                               'artifact_correction': args.artifact_correction,
                               'apply_merging': args.apply_merging,
                               'fuse_z_seeds': args.fuse_z_seeds if '3D' in ct else False,
                               'DET (01)': np.nan,
                               'DET (02)': np.nan,
                               'DET': np.nan,
                               'SEG (01)': np.nan,
                               'SEG (02)': np.nan,
                               'SEG': np.nan,
                               'OP_CSB (01)': np.nan,
                               'OP_CSB (02)': np.nan,
                               'OP_CSB': np.nan,
                               'SO (01)': np.nan,
                               'SO (02)': np.nan,
                               'SO': np.nan,
                               'FPV (01)': np.nan,
                               'FPV (02)': np.nan,
                               'FPV': np.nan,
                               'FNV (01)': np.nan,
                               'FNV (02)': np.nan,
                               'FNV': np.nan,
                               'mode': args.mode,
                               }
                    for train_set in train_sets:
                        print('Evaluate {} on {}_{}: th_seed: {}, th_cell: {}'.format(model.stem, ct, train_set,
                                                                                      th_seed, th_cell))
                        path_seg_results = path_data / ct / "{}_RES_{}_{}_{}".format(train_set, model.stem, th_seed, th_cell)
                        path_seg_results.mkdir(exist_ok=True)
                        # Overwrite existing results
                        if (path_seg_results / "DET_log.txt").exists():
                            os.remove(path_seg_results / "DET_log.txt")
                        if (path_seg_results / "SEG_log.txt").exists():
                            os.remove(path_seg_results / "SEG_log.txt")

                        # Get post-processing settings
                        eval_args = EvalArgs(th_cell=float(th_cell), th_seed=float(th_seed), n_splitting=args.n_splitting,
                                             apply_clahe=args.apply_clahe, scale=scale_factor, cell_type=ct,
                                             save_raw_pred=args.save_raw_pred,
                                             artifact_correction=args.artifact_correction,
                                             apply_merging=args.apply_merging,
                                             fuse_z_seeds=args.fuse_z_seeds)

                        if '2D' in ct:
                            inference_2d_ctc(model=model,
                                             data_path=path_data / ct / train_set,
                                             result_path=path_seg_results,
                                             device=device,
                                             batchsize=args.batch_size,
                                             args=eval_args,
                                             num_gpus=num_gpus)
                        else:
                            inference_3d_ctc(model=model,
                                             data_path=path_data / ct / train_set,
                                             result_path=path_seg_results,
                                             device=device,
                                             batchsize=args.batch_size,
                                             args=eval_args,
                                             num_gpus=num_gpus)

                        seg_measure, det_measure = ctc_metrics(path_data=path_data / ct,
                                                               path_results=path_seg_results,
                                                               path_software=path_ctc_metric,
                                                               subset=train_set,
                                                               mode=args.mode)

                        # For evaluation on silver truth only the SEG measure is used/calculated
                        if args.mode == 'ST':
                            so, fnv, fpv = np.nan, np.nan, np.nan
                        else:
                            _, so, fnv, fpv = count_det_errors(path_seg_results / "DET_log.txt")

                        results[f'DET ({train_set})'] = det_measure
                        results[f'SEG ({train_set})'] = seg_measure
                        results[f'OP_CSB ({train_set})'] = np.nansum([det_measure, seg_measure]) / 2
                        results[f'SO ({train_set})'] = so
                        results[f'FPV ({train_set})'] = fpv
                        results[f'FNV ({train_set})'] = fnv

                    results['DET'] = np.nansum([results['DET (01)'], results['DET (02)']]) / 2
                    results['SEG'] = np.nansum([results['SEG (01)'], results['SEG (02)']]) / 2
                    results['OP_CSB'] = np.nansum([results['OP_CSB (01)'], results['OP_CSB (02)']]) / 2
                    results['SO'] = np.nansum([results['SO (01)'], results['SO (02)']])
                    results['FPV'] = np.nansum([results['FPV (01)'], results['FPV (02)']])
                    results['FNV'] = np.nansum([results['FNV (01)'], results['FNV (02)']])
                    scores.append(results)

    # Convert to dataframe, merge with existing results and save
    scores_df = pd.DataFrame(scores)
    if (path_best_models.parent / "metrics.csv").is_file():
        old_scores_df = pd.read_csv(path_best_models.parent / "metrics.csv")
        scores_df = pd.concat([scores_df, old_scores_df])
        # Delete duplicate entries
        scores_df = scores_df.drop_duplicates(subset=['model', 'cell type', 'th_seed', 'th_cell', 'n_splitting',
                                                      'apply_clahe', 'scale_factor', 'artifact_correction',
                                                      'apply_merging', 'fuse_z_seeds', 'mode'],
                                              keep='first')
    scores_df = scores_df.sort_values(by=['cell type', 'model'])
    scores_df.to_csv(path_best_models.parent / "metrics.csv", header=True, index=False)

    # Get best model and copy to ./models/best_model
    best_settings = utils.get_best_model(scores_df=scores_df, cell_types=cell_type_list, subset=args.subset)
    for key in best_settings:
        if type(best_settings[key]) == np.bool_:
            best_settings[key] = bool(best_settings[key])
    utils.copy_best_model(path_models=path_models,
                          path_best_models=path_best_models,
                          best_model=best_settings['model'],
                          best_settings=best_settings)

    print(f"Best model for {cell_type_list}: {best_settings['model']}")


if __name__ == "__main__":

    main()
