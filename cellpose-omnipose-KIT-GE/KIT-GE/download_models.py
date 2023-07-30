import os
import requests
import zipfile

from pathlib import Path
from shutil import move, rmtree


def download_data(url, target):

    local_filename = target / url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


if __name__ == "__main__":

    model_path = Path(__file__).parent / 'models' / 'kit-sch-ge'
    model_url = 'http://public.celltrackingchallenge.net/participants/KIT-Sch-GE%20(2).zip'

    # Download models
    if len(list(model_path.glob('*.pth'))) == 0:
        print('Downloading models ...')
        download_data(url=model_url, target=model_path)

        # Unzip models
        print('Unzip models ...')
        with zipfile.ZipFile(model_path / model_url.split('/')[-1], 'r') as z:
            z.extractall('models/kit-sch-ge')

        # Remove zip
        os.remove(model_path / model_url.split('/')[-1])

        # Move models
        for file_name in (model_path / 'KIT-Sch-GE (2)' / 'models').glob('*'):
            move(str(file_name), str(model_path))

        # Delete Software (not needed here)
        rmtree(str(model_path / 'KIT-Sch-GE (2)'))

