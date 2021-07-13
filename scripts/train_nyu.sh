DATA_ROOT=./../depth_data
TRAIN_SET=$DATA_ROOT/nyu/
python train.py $TRAIN_SET \
--folder-type sequence \
--resnet-layers 18 \
--num-scales 1 \
-b8 -s0.1 -c0.5 --epoch-size 0 --epochs 50 \
--rc 1 --rt 1 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset nyu \
--name pseudo_Ranking_Plane


# --pretrained-disp=checkpoints/r18_nyu_stn/07-02-00-26/disp_model_best.pth.tar \
# --pretrained-pose=checkpoints/r18_nyu_stn/07-02-00-26/pose_model_best.pth.tar \
# --pretrained-stn=checkpoints/r18_nyu_stn/07-02-00-26/stn_model_best.pth.tar \

