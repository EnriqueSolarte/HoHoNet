from ast import arg
import os
import glob
import json
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from imageio import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from lib.config import config, update_config, infer_exp_id
import matplotlib.pyplot as plt


def draw_boundary(image, u_bound, color=(0, 512, 0), size=2):
    [cv2.line(image, (i, u_bound[i]), (i+1, u_bound[i+1]), color, size)
     for i in range(u_bound.size-1)]
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cfg', default="/media/NFS/kike/HoHoNet/config/mp3d_layout/mlc_mp3d_fpe_retraining.yaml")
    parser.add_argument(
        '--pth_ref', default='/media/NFS/kike/HoHoNet/ckpt/pretrained/ep300.pth')
    # parser.add_argument('--pth', default='/media/NFS/kike/HoHoNet/ckpt/mp3d_layout_mlc_mp3d_fpe_retraining/ep70.pth')
    parser.add_argument(
        '--pth', default='/media/NFS/kike/RETRAINING_LY/neurips_rebuttal_train/HoHoNet/best_model/ckpt/best_2d_iou.pth')

    # parser.add_argument('--img_glob', default='/media/NFS/kike/HoHoNet/assets/28.png')
    parser.add_argument(
        '--img_glob', default='/media/NFS/justin/retraining/mvl/pruned_test/rgb/*')
    parser.add_argument(
        '--output_dir', default='/media/NFS/kike/RETRAINING_LY/datasets/mlc/neurips_rebuttal/HoHoNet')

    parser.add_argument('--save', action='store_true')

    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=int,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Init setting
    update_config(config, args)
    device = torch.device('cpu' if args.no_cuda else 'cuda:0')

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Prepare the trained model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)

    # # Check target directory
    # if not os.path.isdir(args.output_dir):
    #     print('Output directory %s not existed. Create one.' % args.output_dir)
    #     os.makedirs(args.output_dir)

    # Inferencing
    with torch.no_grad():
        plt.figure(0, dpi=500)

        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            fname = os.path.splitext(os.path.split(i_path)[-1])[0]
            print(fname)
            # Load image
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255]).to(device)

            colors = dict(mlc_the_best=(255, 0, 255), pre_trained=(0, 255, 0))
            img = imread(i_path)
            
            bounds_pred = {}
            for pth, color in zip((args.pth, args.pth_ref), colors.items()):
                net.load_state_dict(torch.load(pth, map_location=device))
                net = net.to(device).eval()

                # Inferenceing corners
                net.fname = k
                if args.save:
                    pred_ = net.forward(x)['bon']
                    bounds_pred[color[0]] = pred_.cpu().numpy().squeeze()
                pred = net.infer(x)
                u_bounds = pred['y_bon_'].astype(int).copy()

                draw_boundary(img, u_bounds[0], color[1], size=2)
                draw_boundary(img, u_bounds[1], color[1], size=2)

            if args.save:
                dir_img = os.path.join(args.output_dir, 'img')
                os.makedirs(dir_img, exist_ok=True)
                imsave(os.path.join(dir_img, f"{fname}.jpg"), img)
                for lb, bound_array in bounds_pred.items():
                    dir_bound = os.path.join(args.output_dir, lb)
                    os.makedirs(dir_bound, exist_ok=True)
                    np.save(os.path.join(dir_bound, f"{fname}.npy"), bound_array)
                
            else:
                plt.clf()
                plt.imshow(img)
                plt.axis("off")
                plt.draw()
                plt.waitforbuttonpress(0.1)
                input("Press any key ... <<<<")
