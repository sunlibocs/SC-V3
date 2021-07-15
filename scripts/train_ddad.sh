DATA_ROOT=./../depth_data
TRAIN_SET=$DATA_ROOT/ddad/training/
python train.py $TRAIN_SET \
--folder-type sequence \
--resnet-layers 18 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --epochs 200 \
--rc 1 --rt 1 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--dataset ddad \
--name baseline_DDAD
