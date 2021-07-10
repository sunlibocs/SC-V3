import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from imageio import imread

import models
from inverse_warp import euler2mat


parser = argparse.ArgumentParser(description='Test STN for rotation estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-stn", type=str, help="pretrained STN path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = parser.parse_args()
  
    stn_net = models.STN().to(device)
    weights = torch.load(args.pretrained_stn)
    stn_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)

    gt_pose = np.genfromtxt(dataset_dir/'pose.txt').astype(np.float32).reshape((-1, 3, 4))
    nframes = gt_pose.shape[0]

    errors = np.zeros((nframes-1, 3), np.float32)

    for idx in tqdm(range(nframes-1)):
        
        # img1 = imread(dataset_dir/'color/{:06d}.jpg'.format(idx))
        # img2 = imread(dataset_dir/'color/{:06d}.jpg'.format(idx+1))
        img1 = imread(dataset_dir/'color/{:06d}.png'.format(idx))
        img2 = imread(dataset_dir/'color/{:06d}.png'.format(idx+1))

        h, w, _ = img1.shape
        if h != args.img_height or w != args.img_width:
            img1 = imresize(img1, (args.img_height, args.img_width)).astype(np.float32)
            img2 = imresize(img2, (args.img_height, args.img_width)).astype(np.float32)
        
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        img1 = ((torch.from_numpy(img1).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
        img2 = ((torch.from_numpy(img2).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
        
        pred_rot = stn_net(img1, img2)
        pred_rot = euler2mat(pred_rot).squeeze(0).cpu().numpy()
    
        gt_rot = np.linalg.inv(gt_pose[idx,:,:3]) @ gt_pose[idx+1,:,:3]

        gt_angle, pred_angle, residual_angle = compute_rot_error(gt_rot, pred_rot)
        errors[idx] = gt_angle, pred_angle, residual_angle

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['gt_angle', 'pred_angle', 'residual_angle']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}, {:10.4f}".format(*std_errors))

    filename = Path(args.output_dir)/'rot_error.txt'
    np.savetxt(filename, errors, delimiter=' ', fmt='%1.8e')


def compute_rot_error(gt_rot, pred_rot):
    def compute_angle(rot_mat):
        R = rot_mat
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        angle = np.arctan2(s, c) / 3.14 * 180
        return angle


    gt_angle = compute_angle(gt_rot)
    pred_angle = compute_angle(pred_rot)
    residual_angle = compute_angle(gt_rot @ pred_rot)

    return gt_angle, pred_angle, residual_angle


if __name__ == '__main__':
    main()
