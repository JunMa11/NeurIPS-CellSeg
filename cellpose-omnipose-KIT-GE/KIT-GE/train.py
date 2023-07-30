import argparse
import hashlib
import numpy as np
import random
import torch
import warnings

from pathlib import Path

from segmentation.training.cell_segmentation_dataset import CellSegDataset
from segmentation.training.autoencoder_dataset import AutoEncoderDataset
from segmentation.training.create_training_sets import create_ctc_training_sets
from segmentation.training.mytransforms import augmentors
from segmentation.training.training import train, train_auto, get_max_epochs, get_weights
from segmentation.training.create_training_sets import get_file
from segmentation.utils import utils, unets

warnings.filterwarnings("ignore", category=UserWarning)


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='KIT-Sch-GE 2021 Cell Segmentation - Training')
    parser.add_argument('--act_fun', '-af', default='relu', type=str, help='Activation function')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='Batch size')
    # parser.add_argument('--cell_type', '-ct', nargs='+', required=True, help='Cell type(s)')
    parser.add_argument('--cell_type', '-ct', default=['BFDP-NeurIPS-Cell'], help='Cell type(s)')
    parser.add_argument('--filters', '-f', nargs=2, type=int, default=[64, 1024], help='Filters for U-net')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of models to train')
    parser.add_argument('--loss', '-l', default='smooth_l1', type=str, help='Loss function')
    parser.add_argument('--mode', '-m', default='GT', type=str, help='Ground truth type / training mode')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--norm_method', '-nm', default='bn', type=str, help='Normalization method')
    parser.add_argument('--optimizer', '-o', default='adam', type=str, help='Optimizer')
    parser.add_argument('--pool_method', '-pm', default='conv', type=str, help='Pool method')
    parser.add_argument('--pre_train', '-pt', default=False, action='store_true', help='Auto-encoder pre-training')
    parser.add_argument('--retrain', '-r', default='', type=str, help='Model to retrain')
    parser.add_argument('--split', '-s', default='01', type=str, help='Train/val split')
    args = parser.parse_args()

    # Paths
    path_data = Path(__file__).parent / 'training_data'
    path_models = Path(__file__).parent / 'models' / 'all'

    # Set device for using CPU or GPU
    device, num_gpus = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()

    # Check if dataset consists in training_data folder
    if len(args.cell_type) > 1:
        es = 0
        for cell_type in args.cell_type:
            if not (path_data / cell_type).exists():
                print('No data for cell type "{}" found in {}'.format(cell_type, path_data))
                es = 1
        if es == 1:
            return
        trainset_name = hashlib.sha1(str(args.cell_type).encode("UTF-8")).hexdigest()[:10]
    else:
        if not (args.cell_type[0] == 'all') and not (path_data / args.cell_type[0]).exists():
            print('No data for cell type "{}" found in {}'.format(args.cell_type[0], path_data))
            return
        trainset_name = args.cell_type[0]

    # Create training sets
    print('Create training sets for {} ...'.format(args.cell_type))
    if args.mode == 'GT+ST':
        for mode in ['GT', 'ST', 'GT+ST']:  # create needed GT & ST sets first
            create_ctc_training_sets(path_data=path_data, mode=mode, cell_type_list=args.cell_type, split=args.split)
    else:
        create_ctc_training_sets(path_data=path_data, mode=args.mode, cell_type_list=args.cell_type, split=args.split)

    # Get model names and how many iterations/models need to be trained
    if trainset_name == "all":
        model_name = 'all{}_{}_model'.format(args.mode, args.split)
    else:
        model_name = '{}_{}_{}_model'.format(trainset_name, args.mode, args.split)

    # Train multiple models
    for i in range(args.iterations):

        run_name = utils.unique_path(path_models, model_name + '_{:02d}.pth').stem

        # Get CNN (double encoder U-Net)
        train_configs = {'architecture': ("DU", args.pool_method, args.act_fun, args.norm_method, args.filters),
                         'batch_size': args.batch_size,
                         'batch_size_auto': 2,
                         'label_type': "distance",
                         'loss': args.loss,
                         'num_gpus': num_gpus,
                         'optimizer': args.optimizer,
                         'run_name': run_name
                         }
        net = unets.build_unet(unet_type=train_configs['architecture'][0],
                               act_fun=train_configs['architecture'][2],
                               pool_method=train_configs['architecture'][1],
                               normalization=train_configs['architecture'][3],
                               device=device,
                               num_gpus=num_gpus,
                               ch_in=1,
                               ch_out=1,
                               filters=train_configs['architecture'][4])

        if args.pre_train and args.retrain:
            raise Exception('Use either the pre-train option --pre_train or the retrain option --retrain')

        if args.retrain:

            old_model = Path(__file__).parent / args.retrain
            if get_file(old_model.parent / "{}.json".format(old_model.stem))['architecture'][-1] != train_configs['architecture'][-1]:
                raise Exception('Architecture of model to retrain does not match.')
            # Get weights of trained model to retrain
            print("Load models of {}".format(old_model.stem))
            net = get_weights(net=net, weights=str('{}.pth'.format(old_model)), num_gpus=num_gpus, device=device)
            train_configs['retrain_model'] = old_model.stem

        # Pre-training of the Encoder in autoencoder style
        train_configs['pre_trained'] = False
        if args.pre_train:

            if args.mode != 'GT' or len(args.cell_type) > 1:
                raise Exception('Pre-training only for GTs and for single cell type')

            # Get CNN (U-Net without skip connections)
            net_auto = unets.build_unet(unet_type='AutoU',
                                        act_fun=train_configs['architecture'][2],
                                        pool_method=train_configs['architecture'][1],
                                        normalization=train_configs['architecture'][3],
                                        device=device,
                                        num_gpus=num_gpus,
                                        ch_in=1,
                                        ch_out=1,
                                        filters=train_configs['architecture'][4])

            # Load training and validation set
            data_transforms_auto = augmentors(label_type='auto', min_value=0, max_value=65535)
            datasets = AutoEncoderDataset(data_dir=path_data / args.cell_type[0],
                                          train_dir=path_data / "{}_{}_{}".format(trainset_name, args.mode, args.split),
                                          transform=data_transforms_auto)

            # Train model
            train_auto(net=net_auto, dataset=datasets, configs=train_configs, device=device,  path_models=path_models)

            # Load best weights and load best weights into encoder
            net_auto = get_weights(net=net_auto, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus, device=device)
            pretrained_dict, net_dict = net_auto.state_dict(), net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}  # 1. filter unnecessary keys
            net_dict.update(pretrained_dict)  # 2. overwrite entries
            net.load_state_dict(net_dict)  # 3. load the new state dict
            train_configs['pre_trained'] = True
            del net_auto

        # Load training and validation set
        data_transforms = augmentors(label_type=train_configs['label_type'], min_value=0, max_value=65535)
        train_configs['data_transforms'] = str(data_transforms)
        dataset_name = "{}_{}_{}".format(trainset_name, args.mode, args.split)
        if trainset_name == 'all':
            if args.mode == 'GT+ST':
                dataset_name = "allGT+allST_{}".format(args.split)
            else:
                dataset_name = "{}{}_{}".format(trainset_name, args.mode, args.split)
        datasets = {x: CellSegDataset(root_dir=path_data / dataset_name, mode=x, transform=data_transforms[x])
                    for x in ['train', 'val']}

        # Get number of training epochs depending on dataset size (just roughly to decrease training time):
        train_configs['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']))

        # Train model
        best_loss = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models)

        # Fine-tune with cosine annealing for Ranger models
        if train_configs['optimizer'] == 'ranger':
            net = unets.build_unet(unet_type=train_configs['architecture'][0],
                                   act_fun=train_configs['architecture'][2],
                                   pool_method=train_configs['architecture'][1],
                                   normalization=train_configs['architecture'][3],
                                   device=device,
                                   num_gpus=num_gpus,
                                   ch_in=1,
                                   ch_out=1,
                                   filters=train_configs['architecture'][4])

            # Get best weights as starting point
            net = get_weights(net=net, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus, device=device)
            # Train further
            _ = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models, best_loss=best_loss)

        # Write information to json-file
        utils.write_train_info(configs=train_configs, path=path_models)


if __name__ == "__main__":

    main()
