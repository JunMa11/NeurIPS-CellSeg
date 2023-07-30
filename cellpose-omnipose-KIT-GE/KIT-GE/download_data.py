import os
import requests
import zipfile

from pathlib import Path


def download_data(url, target):

    local_filename = target / url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


if __name__ == "__main__":

    traindata_path = Path(__file__).parent / 'training_data'
    challengedata_path = Path(__file__).parent / 'challenge_data'
    evalsoftware_path = Path(__file__).parent / 'evaluation_software'

    trainingdata_url = 'http://data.celltrackingchallenge.net/training-datasets/'
    challengedata_url = 'http://data.celltrackingchallenge.net/challenge-datasets/'
    evalsoftware_url = 'http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip'

    cell_types = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-C3DH-A549",
                  "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "Fluo-N3DH-CE",
                  "Fluo-N3DH-CHO", "PhC-C2DH-U373", "PhC-C2DL-PSC", "Fluo-C3DH-A549-SIM", "Fluo-N2DH-SIM+",
                  "Fluo-N3DH-SIM+"]

    for cell_type in cell_types:

        # Download training set
        if not (traindata_path / cell_type).is_dir():
            print('Downloading {} training set ...'.format(cell_type))
            download_data(url="{}{}.zip".format(trainingdata_url, cell_type), target=traindata_path)

            # Unzip training set
            print('Unzip {} training set ...'.format(cell_type))
            with zipfile.ZipFile(traindata_path / "{}.zip".format(cell_type), 'r') as z:
                z.extractall('training_data')

            # Remove zip
            os.remove(traindata_path / "{}.zip".format(cell_type))

        # Download challenge set
        if not (challengedata_path / cell_type).is_dir():
            print('Downloading {} challenge set ...'.format(cell_type))
            download_data(url="{}{}.zip".format(challengedata_url, cell_type), target=challengedata_path)

            # Unzip challenge set
            print('Unzip {} challenge set ...'.format(cell_type))
            with zipfile.ZipFile(challengedata_path / "{}.zip".format(cell_type), 'r') as z:
                z.extractall('challenge_data')

            # Remove zip
            os.remove(challengedata_path / "{}.zip".format(cell_type))

    # Download evaluation software
    if len(list(evalsoftware_path.glob('*'))) <= 1:
        print('Downloading evaluation software ...')
        download_data(url=evalsoftware_url, target=evalsoftware_path)

        # Unzip evaluation software
        print('Unzip evaluation software ...')
        with zipfile.ZipFile(evalsoftware_path / evalsoftware_url.split('/')[-1], 'r') as z:
            z.extractall('evaluation_software')

        # Remove zip
        os.remove(evalsoftware_path / evalsoftware_url.split('/')[-1])

