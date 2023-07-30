import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ctc_metrics(path_data, path_results, path_software, subset, mode='GT'):
    """ Cell Tracking Challenge detection and segmentation metrics (DET, SEG).

    :param path_data: Path to directory containing the ground truth.
        :type path_data: pathlib Path object
    :param path_results: Path to directory containing the results (in dirs '01_RES' and '02_RES').
        :type path_results: pathlib Path object
    :param path_software: Path to the evaluation software.
        :type path_software: pathlib Path object
    :param subset: Subset to evaluate ('01_RES', '02_RES')
    :param mode: 'GT' or 'ST' (use STs to calculate the metrics).
    :return: SEG score, DET score.
    """

    if path_data.stem == 'BF-C2DL-HSC' or path_data.stem == 'BF-C2DL-MuSC':
        t = '4'
    else:
        t = '3'

    # Clear temporary result directory if exists
    if os.path.exists(str(path_results.parent / 'tmp')):
        shutil.rmtree(str(path_results.parent / 'tmp'))

    # Copy GT / ST
    (path_results.parent / 'tmp').mkdir(exist_ok=True)
    shutil.copytree(str(path_data / "{}_{}".format(subset, mode)),
                    str(path_results.parent / 'tmp' / "{}_GT".format(subset)))
    shutil.copytree(str(path_results),
                    str(path_results.parent / 'tmp' / "{}_RES".format(subset)))

    # Chose the executable in dependency of the operating system
    if platform.system() == 'Linux':
        path_seg_executable = path_software / 'Linux' / 'SEGMeasure'
        path_det_executable = path_software / 'Linux' / 'DETMeasure'
    elif platform.system() == 'Windows':
        path_seg_executable = path_software / 'Win' / 'SEGMeasure.exe'
        path_det_executable = path_software / 'Win' / 'DETMeasure.exe'
    elif platform.system() == 'Darwin':
        path_seg_executable = path_software / 'Mac' / 'SEGMeasure'
        path_det_executable = path_software / 'Mac' / 'DETMeasure'
    else:
        raise ValueError('Platform not supported')

    # Apply the evaluation software to calculate the cell tracking challenge SEG measure
    output = subprocess.Popen([str(path_seg_executable), str(path_results.parent / 'tmp'), subset, t],
                              stdout=subprocess.PIPE)
    result, _ = output.communicate()
    seg_measure = re.findall(r'\d\.\d*', result.decode('utf-8'))
    seg_measure = float(seg_measure[0])
    shutil.copyfile(str(path_results.parent / 'tmp' / "{}_RES".format(subset) / 'SEG_log.txt'),
                    str(path_results / 'SEG_log.txt'))

    if mode == 'GT':
        output = subprocess.Popen([str(path_det_executable), str(path_results.parent / 'tmp'), subset, t],
                                  stdout=subprocess.PIPE)
        result, _ = output.communicate()
        det_measure = re.findall(r'\d\.\d*', result.decode('utf-8'))
        det_measure = float(det_measure[0])

        # Copy result files
        shutil.copyfile(str(path_results.parent / 'tmp' / "{}_RES".format(subset) / 'DET_log.txt'),
                        str(path_results / 'DET_log.txt'))
    else:
        det_measure = np.nan

    # Remove temporary directory
    shutil.rmtree(str(path_results.parent / 'tmp'))

    return seg_measure, det_measure


def count_det_errors(path_det_file):
    """ Count FPs, FNs and needed splitting operations (in the DET metric).

    :param path_det_file: Path to the DET metric result file.
        :type path_det_file: pathlib Path object.
    :return: DET score, splitting operations, false negatives, false positives.
    """

    det_file = open(str(path_det_file))

    count_so, count_fnv, count_fpv = False, False, False
    so, fnv, fpv = 0, 0, 0

    # Count (FP may include split objects)
    read_file = True
    while read_file:
        det_file_line = det_file.readline()
        if 'Splitting Operations' in det_file_line:
            count_so, count_fnv, count_fpv = True, False, False
            continue
        if 'False Negative Vertices' in det_file_line:
            count_so, count_fnv, count_fpv = False, True, False
            continue
        if 'False Positive Vertices' in det_file_line:
            count_so, count_fnv, count_fpv = False, False, True
            continue
        if '===' in det_file_line:
            count_so, count_fnv, count_fpv = False, False, False
            continue
        if 'DET' in det_file_line:
            det = float(det_file_line.split('DET measure: ')[-1].split('\n')[0])
            read_file = False
            det_file.close()

        if count_so:
            so += 1
        if count_fnv:
            fnv += 1
        if count_fpv:
            fpv += 1

    return det, so, fnv, fpv