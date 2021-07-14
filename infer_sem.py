import os
import sys
import time
import glob
import argparse
import importlib
from tqdm import tqdm
from imageio import imread, imwrite
import torch
import numpy as np
import matplotlib.pyplot as plt

from lib.config import config, update_config

COLOR_MAP = np.array([
    [0,   0,   0],
    [255,   0,  40],
    [255,  72,   0],
    [255, 185,   0],
    [205, 255,   0],
    [91, 255,   0],
    [0, 255,  21],
    [0, 255, 139],
    [0, 255, 252],
    [0, 143, 255],
    [0,  23, 255],
    [90,   0, 255],
    [204,   0, 255],
    [255,   0, 191]], dtype=np.uint8)

if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--inp', required=True)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    config.defrost()
    config.merge_from_file(args.cfg)
    config.model.kwargs.modalities_config.SemanticSegmenter.label_weight = ''
    config.freeze()

    device = 'cuda' if config.cuda else 'cpu'
    device = 'cpu'
    # Parse input paths
    rgb_lst = glob.glob(args.inp)
    if len(rgb_lst) == 0:
        print('No images found')
        import sys
        sys.exit()


    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net = net.eval().to(device)

    # Run inference
    with torch.no_grad():
        for path in tqdm(rgb_lst):
            rgb = imread(path)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.
            if x.shape[2:] != config.dataset.common_kwargs.hw:
                x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode='area')
            x = x.to(device)
            ts = time.time()
            pred_sem = net.infer(x)['sem']
            print(f'Eps time: {time.time() - ts:.2f} sec.')
            # if not torch.is_tensor(pred_sem):
            #     pred_depth = pred_depth.pop('depth')

            fname = os.path.splitext(os.path.split(path)[1])[0]
            # imwrite(
            #     os.path.join(args.out, f'{fname}.sem.png'),
            #     COLOR_MAP[pred_sem.squeeze().cpu().numpy().argmax(0)[160:-160]].astype(np.uint16)
            # )

            # plt.figure(figsize=(15,6))

            plt.imshow(COLOR_MAP[pred_sem.squeeze().cpu().numpy().argmax(0)[160:-160]])
            plt.axis('off')
            plt.savefig(
                os.path.join(args.out, f'{fname}.sem.png')
            )
            # plt.title('semantic prediction')
            # plt.show()
