import os
import argparse
import importlib
from tqdm import tqdm, trange
from collections import Counter
import pathlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from lib.config import config, update_config, infer_exp_id
from lib import dataset
import shutil


def train_loop(net, loader, optimizer):
    net.train()
    if config.training.fix_encoder_bn:
        apply_fn_based_on_key(net.encoder, ['bn'], lambda m: m.eval())
    epoch_losses = Counter()
    for iit, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
        # Move data to the given computation device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        # feed forward & compute losses
        losses = net.compute_mlc_bon_losses(batch)
        if len(losses) == 0:
            continue

        # backprop
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Log
        BS = len(batch['x'])
        epoch_losses['N'] += BS
        for k, v in losses.items():
            if torch.is_tensor(v):
                epoch_losses[k] += BS * v.item()
            else:
                epoch_losses[k] += BS * v

    # Statistic over the epoch
    N = epoch_losses.pop('N')
    for k, v in epoch_losses.items():
        epoch_losses[k] = v / N

    return epoch_losses


def setting_up_HN():
    sys.path.append(
        '/media/NFS/kike/RETRAINING_LY/360-retraining/models/HorizonNet')
    sys.path.append('/media/NFS/kike/RETRAINING_LY/360-retraining')


def valid_loop(net, loader):
    setting_up_HN()
    from models.HorizonNet.train_ours import eval_metrics

    net.eval()
    epoch_losses = Counter()
    with torch.no_grad():
        dict_eval = {"2DIoU": [], "3DIoU": []}
        for iit, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # feed forward & compute losses
            losses = net.compute_mlc_bon_losses(batch)
            pred_ = net.forward(batch['x'])['bon']

            try:
                eval_metrics(batch['bon'].cpu().numpy(),
                             pred_.cpu().numpy(), dict_eval)
            except:
                print("frame skipped")
            # Log
            for k, v in losses.items():
                if torch.is_tensor(v):
                    epoch_losses[k] += float(v.item()) / len(loader)
                else:
                    epoch_losses[k] += v / len(loader)

    return epoch_losses, dict_eval


def apply_fn_based_on_key(net, key_lst, fn):
    for name, m in net.named_modules():
        if any(k in name for k in key_lst):
            fn(m)


def group_parameters(net, wd_group_mode):
    wd = []
    nowd = []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if wd_group_mode == 'bn and bias':
            if 'bn' in name or 'bias' in name:
                nowd.append(p)
            else:
                wd.append(p)
        elif wd_group_mode == 'encoder decoder':
            if 'feature_extractor' in name:
                nowd.append(p)
            else:
                wd.append(p)
    return [{'params': wd}, {'params': nowd, 'weight_decay': 0}]


if __name__ == '__main__':

    # Parse args & config
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--cfg', default="/media/NFS/kike/HoHoNet/config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml")
    parser.add_argument(
        '--cfg', default="/media/NFS/kike/HoHoNet/config/fine_tuning_ly/mlc_cfg.yaml")
    # parser.add_argument('--cfg', required=True)
    parser.add_argument(
        '--pth', default='/media/NFS/kike/HoHoNet/ckpt/pretrained/ep300.pth')

    parser.add_argument(
        '--exp', default='test')
    
    parser.add_argument(
        '--output_dir', default='/media/NFS/kike/RETRAINING_LY/neurips_rebuttal_train/HoHoNet')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    # Init global variable
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    output_exp = os.path.join(args.output_dir, str(args.exp))
    if os.path.exists(output_exp):
        shutil.rmtree(output_exp, ignore_errors=True)
    os.makedirs(os.path.join(output_exp, 'log'), exist_ok=True)
    os.makedirs(os.path.join(output_exp, 'ckpt'), exist_ok=True)
    
    device = f'cuda:{config.gpu}' if config.cuda else 'cpu'
    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = True

    # Init dataset5
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.train_kwargs.update(config.dataset.common_kwargs)
    config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    train_dataset = DatasetClass(**config.dataset.train_kwargs)
    valid_dataset = DatasetClass(**config.dataset.valid_kwargs)
    train_loader = DataLoader(train_dataset, config.training.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda,
                              worker_init_fn=lambda x: np.random.seed())
    valid_loader = DataLoader(valid_dataset, 1,
                              num_workers=config.num_workers,
                              pin_memory=config.cuda)

    # Init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)

    net.load_state_dict(torch.load(args.pth, map_location=device))

    if config.training.fix_encoder_bn:
        apply_fn_based_on_key(
            net.encoder, ['bn'], lambda m: m.requires_grad_(False))

    # Init optimizer
    if config.training.optim == 'Adam':
        optimizer = torch.optim.Adam(
            group_parameters(net, config.training.wd_group_mode),
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)
    elif config.training.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            group_parameters(net, config.training.wd_group_mode),
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)
    elif config.training.optim == 'SGD':
        optimizer = torch.optim.SGD(
            group_parameters(net, config.training.wd_group_mode), momentum=0.9,
            lr=config.training.optim_lr, weight_decay=config.training.weight_decay)

    if config.training.optim_poly_gamma > 0:
        def lr_poly_rate(epoch):
            return (1 - epoch / config.training.epoch) ** config.training.optim_poly_gamma
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_poly_rate)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(p * config.training.epoch)
                                   for p in config.training.optim_milestons],
            gamma=config.training.optim_gamma)

    #! Create tensorboard for monitoring training
    tb_writer = SummaryWriter(log_dir=os.path.join(output_exp, 'log'))
    # ! save cfg file
    shutil.copy(dst=os.path.join(output_exp, "cfg.yaml"), src=args.cfg)
    # Start training
    best_2d_iou = 0
    best_3d_iou = 0
    for iep in trange(1, config.training.epoch + 1, position=0):
        print(f"\tExperiment @ >> {output_exp} ")
        # Valid phase
        epoch_losses, eval_metric = valid_loop(net, valid_loader)
        print(f'EP[{iep}/{config.training.epoch}] valid:  ' +
              ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))

        print(f'EP[{iep}/{config.training.epoch}] valid:  ' +
              ' \ '.join([f'{k} {np.mean(v):.4f}' for k, v in eval_metric.items()]))

        [tb_writer.add_scalar(f"valid/{k}", v, iep)
         for k, v in epoch_losses.items()]
        [tb_writer.add_scalar(f"valid/{k}", np.mean(v), iep)
         for k, v in eval_metric.items()]

        current_2d_iou = np.mean(eval_metric['2DIoU'])
        current_3d_iou = np.mean(eval_metric['3DIoU'])

        if best_2d_iou < current_2d_iou:
            # ! Save current Model fro 2DIoU
            best_2d_iou = current_2d_iou
            torch.save(net.state_dict(), os.path.join(
                output_exp, 'ckpt', f'best_2d_iou.pth'))
            print('Model saved best_2d_iou.pth')

        if best_3d_iou < current_3d_iou:
            # ! Save current Model for 3DIoU
            best_3d_iou = current_3d_iou
            torch.save(net.state_dict(), os.path.join(
                output_exp, 'ckpt', f'best_3d_iou.pth'))
            print('Model saved best_3d_iou.pth')

        # Train phase
        epoch_losses = train_loop(net, train_loader, optimizer)
        scheduler.step()
        print(f'EP[{iep}/{config.training.epoch}] train:  ' +
              ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))

        [tb_writer.add_scalar(f"train/{k}", v, iep)
         for k, v in epoch_losses.items()]

        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iep)
        # Periodically save model
        # if iep % config.training.save_every == 0:
        #     torch.save(net.state_dict(), os.path.join(
        #         exp_ckpt_root, f'ep{iep}.pth'))
        #     print('Model saved')

        # # Valid phase
        # epoch_losses = valid_loop(net, valid_loader)
        # print(f'EP[{iep}/{config.training.epoch}] valid:  ' +
        #       ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))
