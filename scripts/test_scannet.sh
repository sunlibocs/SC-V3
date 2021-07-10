# DISPNET=checkpoints/rectified_nyu_r18/04-16-20:10/dispnet_model_best.pth.tar
# DISPNET=checkpoints/r18_nyu_stn/07-02-00-26/disp_model_best.pth.tar

# DATA_ROOT=/media/bjw/Disk/Dataset/scannet_test
# RESULTS_DIR=results/scannet_test/

# # #  test 256*320 images
# # python test_disp.py --dataset scannet --resnet-layers 18 --img-height 256 --img-width 320 \
# # --pretrained-dispnet $DISPNET --dataset-dir $DATA_ROOT/color \
# # --output-dir $RESULTS_DIR

# # evaluate
# python eval_depth.py \
# --dataset scannet \
# --pred_depth=$RESULTS_DIR/predictions.npy \
# --gt_depth=$DATA_ROOT/depth/ \
# --img_dir $DATA_ROOT/color --vis_dir $RESULTS_DIR


DISPNET=checkpoints/7scene_edgeranking/fire/05-01-21:43/disp_model_best.pth.tar

DATA_ROOT=/media/bjw/Disk/Dataset/7scene_vo/fire/seq-03
RESULTS_DIR=results/scene/

#  test 256*320 images
python test_disp.py --dataset scannet --resnet-layers 18 --img-height 256 --img-width 320 \
--pretrained-dispnet $DISPNET --dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR

# evaluate
python eval_depth.py \
--dataset scannet \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth/ \
--img_dir $DATA_ROOT/color --vis_dir $RESULTS_DIR