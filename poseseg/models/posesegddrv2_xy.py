from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F


# 坐标可求导模式
@MODELS.register_module()
class PoseSegDDRNetV2XY(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.num_joints = 33
        decoder_channels = 64
        self.final_heatmap_layer_1 = nn.Sequential(
            nn.Conv2d(decoder_channels+2, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.num_joints, kernel_size=3, stride=1, padding=1
            )
        )
        self.final_heatmap_layer_2 = nn.Sequential(
            nn.Conv2d(decoder_channels+2, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.num_joints, kernel_size=3, stride=1, padding=1
            )
        )

        self.final_seg_layer_1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, 2, kernel_size=1, stride=1, padding=0
            )
        )
        self.final_seg_layer_2 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, 2, kernel_size=1, stride=1, padding=0
            )
        )

        # 仅支持ohem=True
        self.ohem = True
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none' if self.ohem else 'mean')
        self.xy_criterion = torch.nn.MSELoss(reduction='none')
        self.weak_xy_criterion = torch.nn.MSELoss()

        self.seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=30.0,
                     reduction='mean'
                )
            )

        grid_y, grid_x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        grid_x = grid_x.reshape(-1)
        self.grid_x = grid_x.repeat(self.num_joints, 1).view(1, self.num_joints, -1)
        grid_y = grid_y.reshape(-1)
        self.grid_y = grid_y.repeat(self.num_joints, 1).view(1, self.num_joints, -1)
        self.skeleton = [
                    [8,6],
                    [6,5],
                    [5,4],
                    [4,0],
                    [0,1],
                    [1,2],
                    [2,3],
                    [3,7],
                    [0,9],
                    [0,10],
                    [9,10],
                    [11,12],
                    [11,13],
                    [12,14],
                    [11,23],
                    [12,24],
                    [23,24],
                    [23,25],
                    [24,26],
                    [25,27],
                    [26,28],
                    [27,29],
                    [27,31],
                    [31,29],
                    [28,32],
                    [28,30],
                    [30,32],
                    [13,15],
                    [14,16],
                    [15,21],
                    [15,19],
                    [15,17],
                    [17,19],
                    [16,22],
                    [16,18],
                    [16,20],
                    [18,20]
                ]
        self.skeleton = torch.from_numpy(np.array(self.skeleton))
        self.interpolate_pos = torch.range(0,1,0.1).to(torch.float32).view((1, -1, 11))
        self.beta = 1000.0

    def forward_train(self, image, heatmap, offset_x, offset_y, heatmap_weight, joints_vis, segments, joints2d, **kwargs):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, output_layout_2 = output_list[:2]

        #---------------------------------------------------#
        heatmap = heatmap[kwargs['has_joints']]
        heatmap_weight = heatmap_weight[kwargs['has_joints']]
        offset_x = offset_x[kwargs['has_joints']]
        offset_y = offset_y[kwargs['has_joints']]
        joints_vis = joints_vis[kwargs['has_joints']]
        joints2d = joints2d[kwargs['has_joints']]
        seg_labels = segments.to(torch.int64)
        seg_labels = seg_labels[kwargs['has_segments']]

        #--------------------layout_1 landmark loss-----------------------#
        layout_1_seg_logits = self.final_seg_layer_1(output_layout_1)
        output_layout_1 = torch.concat([output_layout_1, torch.softmax(layout_1_seg_logits, 1)], 1)
        layout_1_uv_logits = self.final_heatmap_layer_1(output_layout_1)
        # 对于拥有joints标注的样本
        layout_1_uv_logits_supervised = layout_1_uv_logits[kwargs['has_joints']]
        layout_1_loss_hm = self._compute_loss_with_heatmap(layout_1_uv_logits_supervised, heatmap, heatmap_weight, joints_vis)

        # Bxjoints_numxHxW
        layout_1_uv_heatmap = torch.sigmoid(layout_1_uv_logits_supervised)
        batch_size = layout_1_uv_heatmap.shape[0]
        layout_1_uv_heatmap_hat = layout_1_uv_heatmap.view((batch_size, self.num_joints, -1))
        layout_1_uv_heatmap_hat = torch.softmax(layout_1_uv_heatmap_hat * self.beta, -1)
        
        layout_1_u = torch.sum(layout_1_uv_heatmap_hat * self.grid_y.to(layout_1_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        layout_1_v = torch.sum(layout_1_uv_heatmap_hat * self.grid_x.to(layout_1_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        
        layout_1_xy = torch.stack([layout_1_v, layout_1_u], -1)
        layout_1_loss_xy = self.xy_criterion(layout_1_xy, joints2d)
        layout_1_loss_xy = layout_1_loss_xy * torch.unsqueeze(joints_vis, -1)
        layout_1_loss_xy = torch.mean(layout_1_loss_xy)

        # # 对于拥有弱joints标注的样本
        # layout_1_uv_logits_weak_supervised = layout_1_uv_logits[kwargs['has_weak_joints']]
        # layout_1_uv_heatmap = torch.sigmoid(layout_1_uv_logits_weak_supervised)
        # batch_size = layout_1_uv_heatmap.shape[0]
        # layout_1_uv_heatmap_hat = layout_1_uv_heatmap.view((batch_size, self.num_joints, -1))
        # layout_1_uv_heatmap_hat = torch.softmax(layout_1_uv_heatmap_hat, -1)
        
        # layout_1_u = torch.sum(layout_1_uv_heatmap_hat * self.grid_y.to(layout_1_uv_heatmap_hat.device), -1, keepdim=False) / 64.0 * 256.0
        # layout_1_v = torch.sum(layout_1_uv_heatmap_hat * self.grid_x.to(layout_1_uv_heatmap_hat.device), -1, keepdim=False) / 64.0 * 256.0
        # # BxN
        # layout_1_skeleton_u_s = torch.index_select(layout_1_u, 1, self.skeleton[:,0].to(layout_1_u.device))
        # layout_1_skeleton_u_s = torch.unsqueeze(layout_1_skeleton_u_s, -1).repeat(1,1,11)
        # layout_1_skeleton_u_e = torch.index_select(layout_1_u, 1, self.skeleton[:,1].to(layout_1_u.device))
        # layout_1_skeleton_u_e = torch.unsqueeze(layout_1_skeleton_u_e, -1).repeat(1,1,11)
        # layout_1_skeleton_v_s = torch.index_select(layout_1_v, 1, self.skeleton[:,0].to(layout_1_v.device))
        # layout_1_skeleton_v_s = torch.unsqueeze(layout_1_skeleton_v_s, -1).repeat(1,1,11)
        # layout_1_skeleton_v_e = torch.index_select(layout_1_v, 1, self.skeleton[:,1].to(layout_1_v.device))
        # layout_1_skeleton_v_e = torch.unsqueeze(layout_1_skeleton_v_e, -1).repeat(1,1,11)

        # # BxNx10
        # layout_1_skeleton_u = torch.lerp(layout_1_skeleton_u_s, layout_1_skeleton_u_e, self.interpolate_pos.to(layout_1_skeleton_u_s.device)) / 256.0 * 2.0 - 1.0
        # layout_1_skeleton_u = layout_1_skeleton_u.reshape(layout_1_skeleton_u.shape[0], -1)
        # BN = layout_1_skeleton_u.shape[0]
        # PN = layout_1_skeleton_u.shape[1]
        # layout_1_skeleton_u = torch.cat([layout_1_skeleton_u, torch.zeros((BN, 256*256-PN)).to(layout_1_skeleton_u.device)], -1)
        # layout_1_skeleton_u = layout_1_skeleton_u.view(BN, 256,256,1)
        # layout_1_skeleton_v = torch.lerp(layout_1_skeleton_v_s, layout_1_skeleton_v_e, self.interpolate_pos.to(layout_1_skeleton_v_s.device)) / 256.0 * 2.0 - 1.0
        # layout_1_skeleton_v = layout_1_skeleton_v.reshape(layout_1_skeleton_v.shape[0], -1)
        # layout_1_skeleton_v = torch.cat([layout_1_skeleton_v, torch.zeros((BN, 256*256-PN)).to(layout_1_skeleton_v.device)], -1)
        # layout_1_skeleton_v = layout_1_skeleton_v.view(BN, 256,256,1)
        # layout_1_skeleton_xy = torch.cat([layout_1_skeleton_v, layout_1_skeleton_u], -1)

        # weak_segment = torch.unsqueeze(segments.to(torch.float32)[kwargs['has_weak_joints']], 1)
        # weak_segment[weak_segment>1] = 0
        # layout_1_skeleton_xy_val = F.grid_sample(weak_segment, layout_1_skeleton_xy, mode='nearest', padding_mode='zeros')
        # layout_1_skeleton_xy_val = layout_1_skeleton_xy_val.view((BN,-1))
        # layout_1_skeleton_xy_val = layout_1_skeleton_xy_val[:,:PN]
        # layout_1_skeleton_xy_loss = self.weak_xy_criterion(layout_1_skeleton_xy_val, torch.ones((BN, PN)).to(layout_1_skeleton_xy_val.device))

        #---------------------layout_1 seg loss---------------------------#
        layout_1_seg_logits = layout_1_seg_logits[kwargs['has_segments']]
        layout_1_seg_logits = F.interpolate(layout_1_seg_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_1_seg_loss_value = self.seg_loss(layout_1_seg_logits, seg_labels, ignore_index=255)

        #--------------------layout_2 landmark loss-----------------------#
        layout_2_seg_logits = self.final_seg_layer_2(output_layout_2)
        output_layout_2 = torch.concat([output_layout_2, torch.softmax(layout_2_seg_logits, 1)], 1)
        layout_2_uv_logits = self.final_heatmap_layer_2(output_layout_2)
        # 对于拥有joints标注的样本
        layout_2_uv_logits_supervised = layout_2_uv_logits[kwargs['has_joints']]
        layout_2_loss_hm = self._compute_loss_with_heatmap(layout_2_uv_logits_supervised, heatmap, heatmap_weight, joints_vis)

        # Bxjoints_numxHxW
        layout_2_uv_heatmap = torch.sigmoid(layout_2_uv_logits_supervised)
        batch_size = layout_2_uv_heatmap.shape[0]
        layout_2_uv_heatmap_hat = layout_2_uv_heatmap.view((batch_size, self.num_joints, -1))
        layout_2_uv_heatmap_hat = torch.softmax(layout_2_uv_heatmap_hat * self.beta, -1)
        
        layout_2_u = torch.sum(layout_2_uv_heatmap_hat * self.grid_y.to(layout_2_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        layout_2_v = torch.sum(layout_2_uv_heatmap_hat * self.grid_x.to(layout_2_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        
        layout_2_xy = torch.stack([layout_2_v, layout_2_u], -1)
        layout_2_loss_xy = self.xy_criterion(layout_2_xy, joints2d)
        layout_2_loss_xy = layout_2_loss_xy * torch.unsqueeze(joints_vis, -1)
        layout_2_loss_xy = torch.mean(layout_2_loss_xy)

        # # 对于拥有弱joints标注的样本
        # layout_2_uv_logits_weak_supervised = layout_2_uv_logits[kwargs['has_weak_joints']]
        # layout_2_uv_heatmap = torch.sigmoid(layout_2_uv_logits_weak_supervised)
        # batch_size = layout_2_uv_heatmap.shape[0]
        # layout_2_uv_heatmap_hat = layout_2_uv_heatmap.view((batch_size, self.num_joints, -1))
        # layout_2_uv_heatmap_hat = torch.softmax(layout_2_uv_heatmap_hat, -1)
        
        # layout_2_u = torch.sum(layout_2_uv_heatmap_hat * self.grid_y.to(layout_2_uv_heatmap_hat.device), -1, keepdim=False) / 64.0 * 256.0
        # layout_2_v = torch.sum(layout_2_uv_heatmap_hat * self.grid_x.to(layout_2_uv_heatmap_hat.device), -1, keepdim=False) / 64.0 * 256.0
        # # BxN
        # layout_2_skeleton_u_s = torch.index_select(layout_2_u, 1, self.skeleton[:,0].to(layout_2_u.device))
        # layout_2_skeleton_u_s = torch.unsqueeze(layout_2_skeleton_u_s, -1).repeat(1,1,11)
        # layout_2_skeleton_u_e = torch.index_select(layout_2_u, 1, self.skeleton[:,1].to(layout_2_u.device))
        # layout_2_skeleton_u_e = torch.unsqueeze(layout_2_skeleton_u_e, -1).repeat(1,1,11)
        # layout_2_skeleton_v_s = torch.index_select(layout_2_v, 1, self.skeleton[:,0].to(layout_2_v.device))
        # layout_2_skeleton_v_s = torch.unsqueeze(layout_2_skeleton_v_s, -1).repeat(1,1,11)
        # layout_2_skeleton_v_e = torch.index_select(layout_2_v, 1, self.skeleton[:,1].to(layout_2_v.device))
        # layout_2_skeleton_v_e = torch.unsqueeze(layout_2_skeleton_v_e, -1).repeat(1,1,11)

        # # BxNx10
        # layout_2_skeleton_u = torch.lerp(layout_2_skeleton_u_s, layout_2_skeleton_u_e, self.interpolate_pos.to(layout_2_skeleton_u_s.device)) / 256.0 * 2.0 - 1.0
        # layout_2_skeleton_u = layout_2_skeleton_u.reshape(layout_2_skeleton_u.shape[0], -1)
        # BN = layout_2_skeleton_u.shape[0]
        # PN = layout_2_skeleton_u.shape[1]
        # layout_2_skeleton_u = torch.cat([layout_2_skeleton_u, torch.zeros((BN, 256*256-PN)).to(layout_2_skeleton_u.device)], -1)
        # layout_2_skeleton_u = layout_2_skeleton_u.view(BN, 256,256,1)
        # layout_2_skeleton_v = torch.lerp(layout_2_skeleton_v_s, layout_2_skeleton_v_e, self.interpolate_pos.to(layout_2_skeleton_v_s.device)) / 256.0 * 2.0 - 1.0
        # layout_2_skeleton_v = layout_2_skeleton_v.reshape(layout_2_skeleton_v.shape[0], -1)
        # layout_2_skeleton_v = torch.cat([layout_2_skeleton_v, torch.zeros((BN, 256*256-PN)).to(layout_2_skeleton_v.device)], -1)
        # layout_2_skeleton_v = layout_2_skeleton_v.view(BN, 256,256,1)
        # layout_2_skeleton_xy = torch.cat([layout_2_skeleton_v, layout_2_skeleton_u], -1)

        # weak_segment = torch.unsqueeze(segments.to(torch.float32)[kwargs['has_weak_joints']], 1)
        # weak_segment[weak_segment>1] = 0
        # layout_2_skeleton_xy_val = F.grid_sample(weak_segment, layout_2_skeleton_xy, mode='nearest', padding_mode='zeros')
        # layout_2_skeleton_xy_val = layout_2_skeleton_xy_val.view((BN,-1))
        # layout_2_skeleton_xy_val = layout_2_skeleton_xy_val[:,:PN]
        # layout_2_skeleton_xy_loss = self.weak_xy_criterion(layout_2_skeleton_xy_val, torch.ones((BN, PN)).to(layout_2_skeleton_xy_val.device))

        #---------------------layout_2 seg loss---------------------------#
        layout_2_seg_logits = layout_2_seg_logits[kwargs['has_segments']]
        layout_2_seg_logits = F.interpolate(layout_2_seg_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_2_seg_loss_value = self.seg_loss(layout_2_seg_logits, seg_labels, ignore_index=255)

        loss_output = {
            'layout_1_loss_uv_hm': layout_1_loss_hm,
            'layout_1_loss_xy': layout_1_loss_xy * 0.5,
            'layout_1_loss_seg': layout_1_seg_loss_value,
            # 'layout_1_loss_skeleton_xy':layout_1_skeleton_xy_loss,
            'layout_2_loss_uv_hm': layout_2_loss_hm,
            'layout_2_loss_xy': layout_2_loss_xy * 0.5,
            'layout_2_loss_seg': layout_2_seg_loss_value,
            # 'layout_2_loss_skeleton_xy': layout_2_skeleton_xy_loss
        }

        return loss_output

    def get_max_pred_batch(self, batch_heatmaps, batch_offset_x, batch_offset_y):
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.max(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        for r in range(batch_size):
            for c in range(num_joints):
                dx = batch_offset_x[r, c, int(preds[r, c, 1]), int(preds[r, c, 0])]
                dy = batch_offset_y[r, c, int(preds[r, c, 1]), int(preds[r, c, 0])]
                preds[r, c, 0] += dx
                preds[r, c, 1] += dy

        return preds, maxvals

    def forward_test(self, image, **kwargs):
        image_h, image_w = image.shape[2:]
        output = self.backbone(image)   # x32,x16,x8,x4
        output = self.head(output)      # uv_heatmap, uv_off
  
        uv_heatmap, uv_off = output
        # convert to probability
        uv_heatmap = torch.sigmoid(uv_heatmap)
        heatmap_h, heatmap_w = uv_heatmap.shape[2:]
        joint_num = uv_heatmap.shape[1]
        offset_x, offset_y = uv_off[:, :joint_num, :, :], uv_off[:, joint_num:, :, :]
        preds = uv_heatmap.detach().cpu().numpy()
        offset_x = offset_x.detach().cpu().numpy()
        offset_y = offset_y.detach().cpu().numpy()

        preds, score = self.get_max_pred_batch(preds, offset_x, offset_y)
        preds[:,:, 0] = preds[:,:, 0] * (image_w/heatmap_w)
        preds[:,:, 1] = preds[:,:, 1] * (image_h/heatmap_h)
        pred_gt_hm = np.concatenate([preds, score], axis=2)
        results = {
            'pred_joints2d': [sample_joint2d for sample_joint2d in pred_gt_hm],
        }
        return results

    def _compute_loss_with_heatmap(self, uv_heatmap, heatmap, heatmap_weight, joints_vis) -> Dict[str, torch.Tensor]:
        """compute loss"""
        batch_size = uv_heatmap.shape[0]
        joints_vis = joints_vis.long()
        joint_mask = joints_vis.reshape(batch_size, -1, 1, 1)

        loss_hm = self.cls_criterion(uv_heatmap, heatmap)
        loss_hm = loss_hm * joint_mask
        loss_hm = loss_hm.mean() * 20.0
        return loss_hm

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        # heatmap, offset, seg = self.head(output) # uv_heatmap, uv_off
        # output = torch.concat([output_list[0], output_list[1]], 1)
        output = output_list[0]

        seg_logits = self.final_seg_layer_1(output)
        seg_pred = torch.softmax(seg_logits, 1)
        output = torch.concat([output, seg_pred], 1)

        layout_1_uv_logits = self.final_heatmap_layer_1(output)
        layout_1_uv_heatmap = torch.sigmoid(layout_1_uv_logits)
        layout_1_uv_heatmap_hat = layout_1_uv_heatmap.view((1, self.num_joints, -1))
        layout_1_uv_heatmap_hat = torch.softmax(layout_1_uv_heatmap_hat * self.beta, -1)
        layout_1_u = torch.sum(layout_1_uv_heatmap_hat * self.grid_y.to(layout_1_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        layout_1_v = torch.sum(layout_1_uv_heatmap_hat * self.grid_x.to(layout_1_uv_heatmap_hat.device), -1) / 64.0 * 256.0
        layout_1_xy = torch.stack([layout_1_v, layout_1_u], -1)

        return layout_1_xy, seg_pred