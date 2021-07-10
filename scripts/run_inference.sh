# INPUT_DIR=/media/bjw/Disk/Dataset/demo_frames/xyz
# OUTPUT_DIR=results/xyz/
# DISP_NET=checkpoints/rectified_nyu_r18/04-16-20:10/dispnet_model_best.pth.tar

# python3 run_inference.py --pretrained $DISP_NET --resnet-layers 18 \
# --dataset-dir $INPUT_DIR --output-dir $OUTPUT_DIR --output-disp


INPUT_DIR=/media/bjw/Disk/Dataset/7scene_vo/fire/seq-03/color
OUTPUT_DIR=/media/bjw/Disk/Dataset/7scene_vo/fire/seq-03/pred_depth
DISP_NET=checkpoints/7scene_edgeranking/fire/05-01-21:43/disp_model_best.pth.tar

python3 run_inference.py --pretrained $DISP_NET --resnet-layers 18 \
--dataset-dir $INPUT_DIR --output-dir $OUTPUT_DIR --output-disp