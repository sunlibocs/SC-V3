from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2, inverse_warp
import numpy as np
import math
from surface_normal import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Image_Info():
    def __init__(self, height, width):
        x_row = np.arange(0, width)
        x = np.tile(x_row, (height, 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        # u_u0 = x - width / 2.0
        y_col = np.arange(0, height)  # y_col = np.arange(0, height)
        y = np.tile(y_col, (width, 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        # v_v0 = y - height / 2.0
        # return u_u0, v_v0
        self.xIndex =  x
        self.yIndex = y

    def getIndex(self):
        return self.xIndex, self.yIndex


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

compute_ssim_loss = SSIM().to(device)


def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode):

    photo_loss = 0
    geometry_loss = 0

    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):

        photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose,
                                                            intrinsics, with_ssim, with_mask, with_auto_mask, padding_mode)
        photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img, tgt_img, ref_depth, tgt_depth, pose_inv,
                                                            intrinsics, with_ssim, with_mask, with_auto_mask, padding_mode)

        photo_loss += (photo_loss1 + photo_loss2)
        geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode):

    ref_img_warped, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0,1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    # masking zero values
    valid_mask = (ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3).float() * (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > mask.numel()*0.1:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img):
    def get_smooth_loss(disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img)

    return loss



def compute_NormalSmooth_loss(tgt_depth, tgt_plane, intrinsics, image_info):
    def get_normalSmooth_loss(normal, plane):
        """
        Computes the smoothness loss for a normal image
        The color image is used for edge-aware smoothness
        The normal and plane should be B C H W
        """
        grad_normal_x = torch.abs(normal[:, :, :, :-1] - normal[:, :, :, 1:])
        grad_normal_y = torch.abs(normal[:, :, :-1, :] - normal[:, :, 1:, :])

        dif_plane_x =  plane[:, :, :, :-1] - plane[:, :, :, 1:]
        dif_plane_y = plane[:, :, :-1, :] - plane[:, :, 1:, :]
        boundary_mask_x = dif_plane_x != 0
        boundary_mask_y = dif_plane_y != 0

        plane[:, :, :, :-1][boundary_mask_x] = 0
        plane[:, :, :-1, :][boundary_mask_y] = 0

        valid_mask = plane != 0
        valid_mask = valid_mask.float().expand(-1, 3, -1, -1)

        grad_normal_x *= valid_mask[:, :, :, :-1]
        grad_normal_y *= valid_mask[:, :, :-1, :]

        return grad_normal_x.mean() + grad_normal_y.mean()

    fx = intrinsics[:, 0:1, 0:1] # fx
    fy = intrinsics[:, 1:2, 1:2] # fy
    u0 = intrinsics[:, 0:1, 2:3]
    v0 = intrinsics[:, 1:2, 2:3]
    b, c, h, w = tgt_depth.shape
    fx = fx.unsqueeze(0).permute(1, 2, 3, 0).expand(-1, c, h, w) # b c h w
    fy = fy.unsqueeze(0).permute(1, 2, 3, 0).expand(-1, c, h, w) # b c h w
    u0 = u0.unsqueeze(0).permute(1, 2, 3, 0).expand(-1, c, h, w) # b c h w
    v0 = v0.unsqueeze(0).permute(1, 2, 3, 0).expand(-1, c, h, w) # b c h w
    xIndex_p, yIndex_p = image_info.getIndex()
    u_u0 = xIndex_p.unsqueeze(0).expand(b, -1, -1, -1) - u0
    v_v0 = yIndex_p.unsqueeze(0).expand(b, -1, -1, -1)  - v0

    tgt_normal = surface_normal_from_depth(tgt_depth, fx, fy, u_u0, v_v0)
    loss = get_normalSmooth_loss(tgt_normal, tgt_plane)

    return loss



def compute_depth_gradient_loss(d_pred, d_gt):

    def gradient_loss(d_pred, d_gt):
        mask = (d_gt > 0.01).float()

        # normalize prediction (inverse depth) to 0-1
        d_pred = 1 / d_pred
        d_pred = (d_pred - d_pred.min()) / d_pred.max()

        d_diff = torch.log(d_pred + 1e-6) - torch.log(d_gt + 1e-6)

        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_mask = torch.mul(mask[:, :, :-2, :], mask[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask[:, :, :, :-2], mask[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + 1e-6

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / N
        
        return gradient_loss
    
    # we compute 4 scales
    loss = 0
    for scale in range(4):
        stride = 2**scale
        loss += gradient_loss(d_pred[:,:,::stride, ::stride], d_gt[:,:,::stride, ::stride])

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    # pred : b c h w
    # gt: b h w
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()
    
    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h,w], mode='nearest').squeeze(1)

    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80
    elif dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10
    elif dataset == 'void':
        crop_mask = gt[0] == gt[0]
        max_depth = 10
    elif dataset == 'ddad':
        crop_mask = gt[0] == gt[0]
        max_depth = 200       
    
    min_depth = 1e-3

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(min_depth, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


# compute ranking loss for the parts outside the mask
# def compute_plane_loss(ref_depth, vnl_plane, intrinsic):
#     if vnl_plane.max() > 0:
#         # b, c, h, w = ref_depth.shape
#         # ## VNL loss
#         # vnl_loss = VNL_Planes_Loss(520.0, 520.0, (h, w), xyz_mode='xyz')
#         # # mask_kp1 = vnl_mask > 125 # 225 -> road plane
#         # # # mask_plane = torch.zeros_like(gt_depth, dtype=torch.uint8)
#         # # mask_plane = 1 * mask_kp1
#         # mask_plane = vnl_plane
#         #
#         # focal_length_x = intrinsic[:,0,0]
#         # focal_length_y = intrinsic[:,1,1]
#         # u0 = intrinsic[:,0,2]
#         # v0 = intrinsic[:,1,2]
#         # loss_vnl = vnl_loss(ref_depth, ref_depth, mask_plane, focal_length_x, focal_length_y, u0, v0)
#
#     else:
#         loss_vnl = torch.tensor(0).float().to(device)
#     return loss_vnl