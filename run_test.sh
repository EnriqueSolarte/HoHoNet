
#!/usr/bin/env bash

IMAGE_FILE="assets/1136.png"

# python3 infer_layout.py --cfg config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml --pth ckpt/mp3d_layout_HOHO_layout_aug_efficienthc_Transen1_resnet34/ep300.pth --out assets/ --inp $IMAGE_FILE

python3 infer_depth.py --cfg config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml --pth ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth --out assets/ --inp $IMAGE_FILE
