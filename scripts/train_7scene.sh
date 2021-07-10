# chess fire heads office pumpkin redkitchen stairs

for SCENE in fire heads office pumpkin redkitchen stairs;

do 
DATA_ROOT=/media/bjw/Disk
TRAIN_SET=$DATA_ROOT/Dataset/7scene_train/$SCENE/
python train.py $TRAIN_SET \
--folder-type sequence \
--resnet-layers 18 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 0 --epochs 5 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset nyu \
--pretrained-disp=checkpoints/r18_nyu_stn/07-02-00-26/disp_model_best.pth.tar \
--pretrained-pose=checkpoints/r18_nyu_stn/07-02-00-26/pose_model_best.pth.tar \
--pretrained-stn=checkpoints/r18_nyu_stn/07-02-00-26/stn_model_best.pth.tar \
--name 7scene/$SCENE
done