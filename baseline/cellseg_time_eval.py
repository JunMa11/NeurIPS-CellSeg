"""
The code was adapted from the MICCAI FLARE Challenge
https://flare22.grand-challenge.org/

The testing images will be evaluated one by one. 
To compensate for the Docker container startup time, we give a time tolerance for the running time. 
https://neurips22-cellseg.grand-challenge.org/metrics/
"""

import os
join = os.path.join
import sys
import shutil
import time
import torch
import argparse
from collections import OrderedDict
from skimage import io 
import tifffile as tif 
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('Segmentation efficiency eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./val-imgs-30/', type=str, help='testing data path')
parser.add_argument('-o','--save_path', default='./val_team_seg', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
args = parser.parse_args()

test_img_path = args.test_img_path
save_path = args.save_path
docker_path = args.docker_folder_path

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        # create temp folers for inference one-by-one
        if os.path.exists(input_temp):
            shutil.rmtree(input_temp)
        if os.path.exists(output_temp):
            shutil.rmtree(output_temp)
        os.makedirs(input_temp)
        os.makedirs(output_temp)
        # load docker and create a new folder to save segmentation results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        # os.system('docker image load < {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        metric = OrderedDict()
        metric['Img Name'] = []
        metric['Real Running Time'] = []
        metric['Rank Running Time'] = []
        # To obtain the running time for each case, we inference the testing case one-by-one
        for case in test_cases:
            shutil.copy(join(test_img_path, case), input_temp)
            if case.endswith('.tif') or case.endswith('.tiff'):
                img = tif.imread(join(input_temp, case))
            else:
                img = io.imread(join(input_temp, case))
            pix_num = img.shape[0] * img.shape[1]
            cmd = 'docker container run --gpus="device=0" -m 28g --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
            print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
            start_time = time.time()
            os.system(cmd)
            real_running_time = time.time() - start_time
            print(f"{case} finished! Inference time: {real_running_time}")
            # save metrics
            metric['Img Name'].append(case)
            metric['Real Running Time'].append(real_running_time)
            if pix_num <= 1000000:
                rank_running_time = np.max([0, real_running_time-10])
            else:
                rank_running_time = np.max([0, real_running_time-10*(pix_num/1000000)])
            metric['Rank Running Time'].append(rank_running_time)
            os.remove(join(input_temp, case))  
            seg_name = case.split('.')[0] + '_label.tiff'
            try:
                os.rename(join(output_temp, seg_name), join(team_outpath, seg_name))
            except:
                print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                print("Wrong segmentation name!!! It should be image_name.split(\'.\')[0] + \'_label.tiff\' ")
        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(join(team_outpath, teamname + '_running_time.csv'), index=False)
        torch.cuda.empty_cache()
        # os.system("docker rmi {}:latest".format(teamname))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)
    except Exception as e:
        print(e)
