STN=checkpoints/r18_nyu_stn/07-02-00-26/stn_model_best.pth.tar
POSE=checkpoints/r18_nyu_stn/07-02-00-26/pose_model_best.pth.tar

# STN=checkpoints/7scene/fire/08-13-14:29/stn_model_best.pth.tar

# DATA_DIR=/media/bjw/Disk/Dataset/nyu_480_10/basement_0001a/

DATA_DIR=/media/bjw/Disk/Dataset/nyu_480_10/office/seq-02/

# #  test 256*320 images
# python test_stn.py --img-height 256 --img-width 320 \
# --pretrained-stn $STN --dataset-dir $DATA_DIR \
# --output-dir $DATA_DIR

# #  test 192*256 images
# python test_stn.py --img-height 192 --img-width 256 \
# --pretrained-stn $STN --dataset-dir $DATA_DIR \
# --output-dir $DATA_DIR

python test_vo_stn.py --img-height 256 --img-width 320 \
--pretrained-stn $STN --pretrained-pose $POSE --dataset-dir $DATA_DIR \
--output-dir $DATA_DIR

