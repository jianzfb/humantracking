from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F
import cv2


@MODELS.register_module()
class PoseSegDDRNetV4(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.num_joints = 33
        decoder_channels = 64

        self.final_heatmap_layer_1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.num_joints, kernel_size=3, stride=1, padding=1
            )
        )

        self.final_offset_layer_1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.num_joints * 2, kernel_size=3, stride=1, padding=1
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
                decoder_channels, 1, kernel_size=1, stride=1, padding=0
            )
        )

        # 仅支持ohem=True
        self.ohem = True
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none' if self.ohem else 'mean')
        self.criterion = torch.nn.MSELoss(reduction='none' if self.ohem else 'mean')

        self.seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=10.0,
                     reduction='mean'
                )
            )

        self.offset_loss_weight = 2.0
        self.heatmap_loss_weight = 10.0
        if train_cfg is not None:
            self.offset_loss_weight = train_cfg.get('offset_loss_weight', 0.1)
            self.heatmap_loss_weight = train_cfg.get('heatmap_loss_weight', 1.0)

    def forward_train(self, image, heatmap, offset_x, offset_y, heatmap_weight, joints_vis, segments, **kwargs):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, _ = output_list[:2]

        #---------------------------------------------------#
        heatmap = heatmap[kwargs['has_joints']]
        heatmap_weight = heatmap_weight[kwargs['has_joints']]
        offset_x = offset_x[kwargs['has_joints']]
        offset_y = offset_y[kwargs['has_joints']]
        joints_vis = joints_vis[kwargs['has_joints']]

        seg_labels = segments.to(torch.int64)
        seg_labels = seg_labels[kwargs['has_segments']]

        #--------------------layout_1 landmark loss-----------------------#
        layout_1_seg_logits = self.final_seg_layer_1(output_layout_1)
        layout_1_uv_heatmap = self.final_heatmap_layer_1(output_layout_1)
        layout_1_uv_off = self.final_offset_layer_1(output_layout_1)

        layout_1_uv_heatmap = layout_1_uv_heatmap[kwargs['has_joints']]
        layout_1_uv_off = layout_1_uv_off[kwargs['has_joints']]

        layout_1_pose_loss_value = \
            self._compute_loss_with_heatmap(
                layout_1_uv_heatmap, layout_1_uv_off, heatmap, heatmap_weight, offset_x, offset_y, joints_vis)

        #---------------------layout_1 seg loss---------------------------#
        layout_1_seg_logits = layout_1_seg_logits[kwargs['has_segments']]
        layout_1_seg_logits = F.interpolate(layout_1_seg_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_1_seg_loss_value = self.seg_loss(layout_1_seg_logits.squeeze(1), seg_labels, ignore_index=255)

        loss_output = {
            'layout_1_loss_uv_hm': layout_1_pose_loss_value['loss_uv_hm'],
            'layout_1_loss_xy_offset': layout_1_pose_loss_value['loss_xy_offset'],
            'layout_1_seg_loss': layout_1_seg_loss_value
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
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, output_layout_2 = output_list[:2]

        output = output_layout_1

        seg_logits = self.final_seg_layer_1(output)
        seg_pred = torch.sigmoid(seg_logits)
        uv_heatmap = self.final_heatmap_layer_1(output)
        uv_off = self.final_offset_layer_1(output)

        uv_heatmap = torch.sigmoid(uv_heatmap)
        uv_heatmap = F.max_pool2d(uv_heatmap, 3, stride=1, padding=(3 - 1) // 2)

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

        seg_pred = F.interpolate(seg_pred, (image_h, image_w), mode='bilinear', align_corners=True)
        results = {
            'pred_joints2d': [sample_joint2d for sample_joint2d in pred_gt_hm],
            'pred_segments': (seg_pred.detach().cpu().numpy() > 0.4).astype(np.uint8)
        }
        return results
    
    def _loss(self, pred_heatmap, pred_offset_xy, gt_heatmap, joint_mask, heatmap_mask, offset_x, offset_y):
        loss_hm = self.cls_criterion(pred_heatmap, gt_heatmap)
        joint_num = pred_heatmap.shape[1]
        loss_hm = loss_hm * joint_mask

        hard_weight = 20
        mid_weight = 10
        easy_weight = 5
        if pred_offset_xy is None:
            bs, joint_num = loss_hm.shape[:2]
            loss_hm = torch.reshape(loss_hm, (bs * joint_num, -1))
            hm_items = loss_hm.mean(dim=-1)
            sortids = torch.argsort(hm_items)
            topids = sortids[: (bs * joint_num // 3)]
            midids = sortids[(bs * joint_num // 3) : (bs * joint_num // 3 * 2)]
            bottomids = sortids[(bs * joint_num // 3 * 2) :]

            loss_hm = (
                (hm_items[topids] * easy_weight).mean()
                + (hm_items[midids] * mid_weight).mean()
                + (hm_items[bottomids] * hard_weight).mean()
            )
            return loss_hm, 0.0, 0.0

        loss_offx = self.criterion(pred_offset_xy[:, :joint_num, :, :].mul(heatmap_mask), offset_x.mul(heatmap_mask))
        loss_offy = self.criterion(pred_offset_xy[:, joint_num:, :, :].mul(heatmap_mask), offset_y.mul(heatmap_mask))

        loss_offx = loss_offx * joint_mask
        loss_offy = loss_offy * joint_mask
        loss_offx = loss_offx.sum()/((heatmap_mask*joint_mask).sum()+1e-6) * 10.0
        loss_offy = loss_offy.sum()/((heatmap_mask*joint_mask).sum()+1e-6) * 10.0

        ######################
        bs, joint_num = loss_hm.shape[:2]
        loss_hm = torch.reshape(loss_hm, (bs * joint_num, -1))
        hm_items = loss_hm.mean(dim=-1)
        sortids = torch.argsort(hm_items)
        topids = sortids[: (bs * joint_num // 3)]
        midids = sortids[(bs * joint_num // 3) : (bs * joint_num // 3 * 2)]
        bottomids = sortids[(bs * joint_num // 3 * 2) :]

        loss_hm = (
            (hm_items[topids] * easy_weight).mean()
            + (hm_items[midids] * mid_weight).mean()
            + (hm_items[bottomids] * hard_weight).mean()
        )
        return loss_hm, loss_offx, loss_offy

    def _compute_loss_with_heatmap(self, uv_heatmap, uv_off_xy, heatmap, heatmap_weight, offset_x, offset_y, joints_vis) -> Dict[str, torch.Tensor]:
        """compute loss"""
        batch_size = uv_heatmap.shape[0]
        joints_vis = joints_vis.long()
        joint_mask = joints_vis.reshape(batch_size, -1, 1, 1)

        # 2d heatmap + depth loass
        loss_hm, loss_offx, loss_offy = self._loss(
            uv_heatmap, uv_off_xy, heatmap, joint_mask, heatmap_weight, offset_x, offset_y
        )

        outloss = dict()
        outloss["loss_uv_hm"] = self.heatmap_loss_weight * loss_hm
        outloss["loss_xy_offset"] = self.offset_loss_weight * (loss_offx + loss_offy)
        return outloss

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1 = output_list[0]

        layout_1_seg_logits = self.final_seg_layer_1(output_layout_1)
        layout_1_seg_pred = torch.sigmoid(layout_1_seg_logits)
        uv_heatmap = self.final_heatmap_layer_1(output_layout_1)
        uv_off = self.final_offset_layer_1(output_layout_1)
        uv_heatmap = torch.sigmoid(uv_heatmap)
        return uv_heatmap, uv_off, layout_1_seg_pred