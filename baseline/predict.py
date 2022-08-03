
import os
join = os.path.join
import argparse
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
from baseline.models.unetr2d import UNETR2D
import time
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetr_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=256, type=int, help='segmentation classes')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name.lower() == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)


    if args.model_name.lower() == 'unetr':
        model = UNETR2D(
            in_channels=3,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)


    if args.model_name.lower() == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=3, 
            out_channels=args.num_class,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

    checkpoint = torch.load(join(args.model_path, 'best_Dice_model.pth'), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))
            
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            test_pred_npy = test_pred_out[0,1].cpu().numpy()
            # convert probability map to binary mask and apply morphological postprocessing
            test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>0.5),16))
            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            
        
if __name__ == "__main__":
    main()





