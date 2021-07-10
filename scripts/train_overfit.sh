SCENE=fire

DATA_ROOT=/media/bjw/Disk
TRAIN_SET=$DATA_ROOT/Dataset/7scene_train/$SCENE/
python train.py $TRAIN_SET \
--folder-type sequence \
--resnet-layers 18 \
--num-scales 1 \
-b8 -s0.1 -c0.5 --epoch-size 0 --epochs 301 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--pretrained-disp=checkpoints/r18_nyu_test_edgeranking/04-30-23:04/disp_model_best.pth.tar \
--pretrained-pose=checkpoints/r18_nyu_test_edgeranking/04-30-23:04/pose_model_best.pth.tar \
--pretrained-stn=checkpoints/r18_nyu_test_edgeranking/04-30-23:04/stn_model_best.pth.tar \
--dataset nyu \
--name 7scene_edgeranking/$SCENE

# --pretrained-disp=checkpoints/r18_nyu_stn/07-02-00-26/disp_model_best.pth.tar \
# --pretrained-pose=checkpoints/r18_nyu_stn/07-02-00-26/pose_model_best.pth.tar \
# --pretrained-stn=checkpoints/r18_nyu_stn/07-02-00-26/stn_model_best.pth.tar \