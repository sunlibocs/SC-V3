# DISP_NET=checkpoints-kitti/r18_kitti_stn_h192_b8/07-13-08:05/disp_model_best.pth.tar
# DISP_NET=checkpoints-kitti/r18_kitti_stn_h192_b8/07-12-03:38/disp_model_best.pth.tar

# DISP_NET=checkpoints-kitti/r18_kitti_stn/07-01-23:00/disp_model_best.pth.tar

DISP_NET=/run/user/1000/gvfs/sftp:host=robotvision2.cs.adelaide.edu.au,user=jwbian/home/jwbian/lb/checkpoints/evlDepthJWB_sigma0.1_kit/05-10-03:06/dispnet_model_best.pth.tar


DATA_ROOT=/media/bjw/Disk/Dataset/kitti_depth_test
RESULTS_DIR=results/kitti_test

# # test
# python test_disp.py --resnet-layers 18 --img-height 192 --img-width 640 \
# --pretrained-dispnet $DISP_NET --dataset-dir $DATA_ROOT/color \
# --output-dir $RESULTS_DIR

# test
python test_disp.py --resnet-layers 18 --img-height 256 --img-width 832 \
--pretrained-dispnet $DISP_NET --dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR


# evaluate
python eval_depth.py \
--dataset kitti \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth