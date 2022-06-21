# NeurIPS-CellSeg
Naive baseline for microscopy image segmentation challenge in NeurIPS 2022



## Preprocessing

Download training data to the `data` folder

Run `python pre_process_3class.py`



## Training

`cd baseline`

Run `python model_training_3class.py --data_path 'path to training data' --batch_size 8`



## Inference

Run

`python predict.py -i input_path -o output_path`

> Your prediction file should have at leat the two arguments: `input_path` and `output_path`. The two arguments are important to establish connections between local folders and docker folders.



## Build Docker

### 1) Preparation

The docker is built on [MONAI](https://hub.docker.com/r/projectmonai/monai)

> docker pull projectmonai/monai

Prepare `Dockerfile`

```dockerfile
FROM projectmonai/monai:latest

WORKDIR /workspace
COPY ./   /workspace
```

Put the inference command in the `predict.sh`

```bash
# !/bin/bash -e
python predict.py -i "/workspace/inputs/"  -o "/workspace/outputs/"
```

> The `input_path` and `output_path` augments should specify the corresponding docker workspace folders rather than local folers, because we will map the local folders to the docker workspace folders when running the docker container.

### 2) Build Docker and make sanity test

The submitted docker will be evaluated by the following command:

```bash
docker container run --gpus "device=0" --name teamname --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/teamname_outputs/:/workspace/outputs/ teamname:latest /bin/bash -c "sh predict.sh"
```

- `--name`: container name during running

- `--rm`: remove container after running
- `-v $PWD/CellSeg_Test/:/workspace/inputs/`: map local image data folder to Docker `workspace/inputs` folder. 
- `-v $PWD/teamname_outputs/:/workspace/outputs/ `: map Docker `workspace/outputs` folder to local folder. The segmentation results will be in `$PWD/teamname_outputs`
- `teamname:latest`: docker image name (should be `teamname`) and its version tag. **The version tag should be `latest`**. Please do not use `v0`, `v1`... as the version tag
- `/bin/bash -c "sh predict.sh"`: start the prediction command. It will load testing images from `workspace/inputs` and save the segmentation results to `workspace/outputs`



Assuming the team name is `baseline`, the Docker build command is 

`docker build -t baseline . `

Test the docker to make sure it works. There should be segmentation resoults in the `baseline_outputs` folder.

```bash
docker container run --gpus "device=0" --name baseline --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/baseline_outputs/:/workspace/outputs/ baseline:latest /bin/bash -c "sh predict.sh"
```

> During the inference, please monitor the GPU memory consumption by `watch nvidia-smi`. The GPU memory consumption should be less than 1500MB. Otherwise, it will run into OOM error on the official evaluation server. We use this hard contrain on the GPU memory  consumption because most biologists do not have powerful GPUs in practice. Thus, the model should be low-resource.



### 3) Save Docker

`docker save baseline | gzip -c > baseline.tar.gz`

Upload the docker to Google drive or Baidu net disk and send the download link to `NeurIPS.CellSeg@gmail.com`. 

> Please **do not** upload the Dodker to dockerhub!





