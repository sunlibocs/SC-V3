DATA_ROOT=/media/bjw/Disk
TRAIN_SET=$DATA_ROOT/Dataset/kitti_256/
python train.py $TRAIN_SET \
--folder-type sequence \
--resnet-layers 18 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 0 --epochs 20 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset kitti \
--name r18_kitti_stn
