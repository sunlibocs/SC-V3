import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class MaskRanking_Loss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8):
        super(MaskRanking_Loss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth

    def generate_target(self, depth, pred, theta=0.02):
        B, C, H, W = depth.shape
        mask_A = torch.rand(C, H, W).cuda()
        mask_A[mask_A >= (1 - self.sample_ratio)] = 1
        mask_A[mask_A < (1 - self.sample_ratio)] = 0
        idx = torch.randperm(mask_A.nelement())
        mask_B = mask_A.view(-1)[idx].view(mask_A.size())
        mask_A = mask_A.repeat(B, 1, 1).view(depth.shape) == 1
        mask_B = mask_B.repeat(B, 1, 1).view(depth.shape) == 1
        za_gt = depth[mask_A]
        zb_gt = depth[mask_B]
        mask_ignoreb = zb_gt > self.filter_depth
        mask_ignorea = za_gt > self.filter_depth
        mask_ignore = mask_ignorea | mask_ignoreb
        za_gt = za_gt[mask_ignore]
        zb_gt = zb_gt[mask_ignore]

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 > 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).cuda()
        target[mask1] = 1
        target[mask2] = -1

        return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target

    def cal_ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        pred_depth = z_A - z_B
        log_loss = torch.mean(torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        squared_loss = torch.mean(pred_depth[target == 0] ** 2)  # if pred depth is not zero adds to loss
        return log_loss + squared_loss

    def forward(self, pred_depth, gt_depth, tgt_valid_weight):
        za, zb, target = self.generate_target(gt_depth, pred_depth)
        total_loss = self.cal_ranking_loss(za, zb, target)

        return total_loss
