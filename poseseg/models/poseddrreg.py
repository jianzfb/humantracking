from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.cnn import ConvModule
import numpy as np
import torch.nn.functional as F
import cv2

class Stage_1(nn.Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        # paf
        self.conv1_CPM_L1 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv2_CPM_L1 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv3_CPM_L1 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv4_CPM_L1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # center
        self.conv1_CPM_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv2_CPM_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv3_CPM_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv4_CPM_L2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.conv1_CPM_L1(x) # branch1
        h1 = self.conv2_CPM_L1(h1)
        h1 = self.conv3_CPM_L1(h1)
        h1 = self.conv4_CPM_L1(h1)

        h2 = self.conv1_CPM_L2(x) # branch2
        h2 = self.conv2_CPM_L2(h2)
        h2 = self.conv3_CPM_L2(h2)
        h2 = self.conv4_CPM_L2(h2)
        return h1, h2

class Stage_x(nn.Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        # paf
        self.conv1_L1 = ConvModule(
                    97, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv2_L1 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv3_L1 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))                          
        self.conv4_L1 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)

        # joints
        self.conv1_L2 = ConvModule(
                    97, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv2_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.conv3_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))                    
        self.conv4_L2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        h1 = self.conv1_L1(x) # branch1
        h1 = self.conv2_L1(h1)
        h1 = self.conv3_L1(h1)
        h1 = self.conv4_L1(h1)

        h2 = self.conv1_L2(x) # branch2
        h2 = self.conv2_L2(h2)
        h2 = self.conv3_L2(h2)
        h2 = self.conv4_L2(h2)
        return h1, h2


@MODELS.register_module()
class PoseDDRReg(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.num_joints = 33

        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()

        self.joint_weight = torch.from_numpy(train_cfg.joint_weight)
        self.limb_weight = torch.from_numpy(train_cfg.limb_weight)

    def forward_train(self, image, heatmaps, pafs, ignore_mask, heatmap_valid_mask, paf_valid_mask, **kwargs):
        # ddr backbone
        output_list = self.backbone(image)   # x32,x16,x8,x4
        _, _, base_and_delta_2 = output_list
        base_and_delta_2 = F.relu(base_and_delta_2)
 
        # 计算特征
        pafs_p = []
        heatmaps_p = []
        h1, h2 = self.stage_1(base_and_delta_2)
        pafs_p.append(h1)
        heatmaps_p.append(h2)

        h1, h2 = self.stage_2(torch.cat([h1,h2,base_and_delta_2], dim=1))
        pafs_p.append(h1)
        heatmaps_p.append(h2)

        h1, h2 = self.stage_3(torch.cat([h1,h2,base_and_delta_2], dim=1))
        pafs_p.append(h1)
        heatmaps_p.append(h2)

        # 计算损失
        paf_g = pafs
        heatmap_g = heatmaps
        ignore_mask_g = ignore_mask
        heatmap_valid_mask_g = heatmap_valid_mask
        paf_valid_mask_g = paf_valid_mask
        total_loss, paf_loss, heatmap_loss = self.compute_loss(pafs_p, heatmaps_p, paf_g, heatmap_g, ignore_mask_g, heatmap_valid_mask_g, paf_valid_mask_g)

        loss_output = {
            'loss': total_loss
        }
        return loss_output

    def compute_loss(self, pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask, heatmap_valid_mask_t, paf_valid_mask_t):
        heatmap_loss_log = []
        paf_loss_log = []
        total_loss = 0

        # paf_masks = ignore_mask.unsqueeze(1).repeat([1, pafs_t.shape[1], 1, 1])
        # heatmap_masks = ignore_mask.unsqueeze(1).repeat([1, heatmaps_t.shape[1], 1, 1])

        # compute loss on each stage
        for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
            if pafs_y.shape != pafs_t.shape:
                pafs_y = F.interpolate(pafs_y, pafs_t.shape[2:], mode='bilinear', align_corners=True)
                heatmaps_y = F.interpolate(heatmaps_y, heatmaps_t.shape[2:], mode='bilinear', align_corners=True)

            # with torch.no_grad():
            #     # 忽略的地方，设置成和预测值相同，无梯度
            #     stage_pafs_t[stage_paf_masks == 1] = pafs_y.detach()[stage_paf_masks == 1]
            #     stage_heatmaps_t[stage_heatmap_masks == 1] = heatmaps_y.detach()[stage_heatmap_masks == 1]        
            if self.limb_weight.device != pafs_y.device:
                self.limb_weight = self.limb_weight.to(pafs_y.device)
            if self.joint_weight.device != heatmaps_y.device:
                self.joint_weight = self.joint_weight.to(heatmaps_y.device)

            pafs_loss = self._loss(pafs_y, pafs_t, paf_valid_mask_t, self.limb_weight)
            heatmaps_loss = self._loss(heatmaps_y, heatmaps_t, heatmap_valid_mask_t, self.joint_weight)
            total_loss += pafs_loss + heatmaps_loss

            paf_loss_log.append(pafs_loss.item())
            heatmap_loss_log.append(heatmaps_loss.item())

        return total_loss, np.array(paf_loss_log), np.array(heatmap_loss_log)

    def _loss(self, pred, target, mask, weight):
        loss = torch.mean(((pred - target)*mask) ** 2, dim=[2,3])
        loss = loss * weight

        hard_weight = 30
        mid_weight = 15
        easy_weight = 5
        bs, joint_num = loss.shape[:2]
        loss = torch.reshape(loss, (bs * joint_num, -1))
        hm_items = loss.mean(dim=-1)
        sortids = torch.argsort(hm_items)
        topids = sortids[: (bs * joint_num // 3)]
        midids = sortids[(bs * joint_num // 3) : (bs * joint_num // 3 * 2)]
        bottomids = sortids[(bs * joint_num // 3 * 2) :]

        loss_hm = (
            (hm_items[topids] * easy_weight).mean()
            + (hm_items[midids] * mid_weight).mean()
            + (hm_items[bottomids] * hard_weight).mean()
        )
        return loss_hm

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        _, _, base_and_delta_2 = output_list
        base_and_delta_2 = F.relu(base_and_delta_2)

        h1, h2 = self.stage_1(base_and_delta_2)
        h1, h2 = self.stage_2(torch.cat([h1,h2,base_and_delta_2], dim=1))
        h1, h2 = self.stage_3(torch.cat([h1,h2,base_and_delta_2], dim=1))
        return h1, h2
