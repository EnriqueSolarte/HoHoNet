

ckpt=ckpt/mp3d_layout_mlc_mp3d_fpe_retraining/ep30.pth

cfg=config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml
img=assets/28.png
output_dir=test/mp3d_fpe/retraining/

python test_layout.py --cfg $cfg --pth $ckpt --img_glob $img --output_dir $output_dir
python vis_layout.py --img assets/28.png --layout test/mp3d_fpe/retraining/28.txt 