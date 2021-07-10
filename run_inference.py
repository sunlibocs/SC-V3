import torch
from imageio import imread, imsave
import torch.nn.functional as F
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import cv2

from models import DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')
parser.add_argument("--save-video", action='store_true', help='save videos using opencv')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        img = imread(file)

        tensor_img = ((torch.from_numpy(np.transpose(img, (2, 0, 1))).float()/255 - 0.45)/0.225).unsqueeze(0).to(device)

        b, c, h, w = tensor_img.size()
        if h != args.img_height or w != args.img_width:
            tensor_img = F.interpolate(tensor_img, (args.img_height, args.img_width), mode='bilinear', align_corners=True)
        
        pred_disp = disp_net(tensor_img)

        if h != args.img_height or w != args.img_width:
            pred_disp = F.interpolate(pred_disp, (h, w), mode='nearest')

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = file_path.basename()

        # if args.output_disp:
        #     disp = (255*tensor2array(pred_disp[0], max_value=10, colormap='bone')).astype(np.uint8)
        #     imsave(output_dir/'{}.png'.format(file_name), np.transpose(disp, (1, 2, 0)))
        # if args.output_depth:
        #     depth = 1/pred_disp
        #     depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
        #     imsave(output_dir/'{}.png'.format(file_name), np.transpose(depth, (1, 2, 0)))

        depth = (0.210 / pred_disp[0,0].cpu().numpy() * 5000)
        imsave(output_dir/'{}.png'.format(file_name), depth.astype(np.uint16))
        
        


if __name__ == '__main__':
    main()
