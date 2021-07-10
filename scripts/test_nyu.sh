#  DISPNET=checkpoints/r18_nyu_10/04-16-07:59/dispnet_model_best.pth.tar
# DISPNET=checkpoints/rectified_nyu_r18/04-16-20:10/dispnet_model_best.pth.tar

# DISPNET=checkpoints/r18_nyu_stn/07-01-18:28/disp_model_best.pth.tar # only rt
# DISPNET=checkpoints/r18_nyu_stn/07-02-00-26/disp_model_best.pth.tar # rt+rc

DISPNET=checkpoints/r18_nyu_test_edgeranking/04-30-23:04/disp_model_best.pth.tar

DATA_ROOT=/media/bjw/Disk/Dataset/nyu_test
RESULTS_DIR=results/nyu_test/

# #  test 256*320 images
# python test_disp.py --dataset nyu --resnet-layers 18 --img-height 256 --img-width 320 \
# --pretrained-dispnet $DISPNET --dataset-dir $DATA_ROOT/color \
# --output-dir $RESULTS_DIR

# evaluate
python eval_depth.py \
--dataset nyu \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth.npy \
--img_dir $DATA_ROOT/color --vis_dir $RESULTS_DIR