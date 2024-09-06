from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.models.builder import build_backbone, build_head, build_neck
import torchvision
import numpy as np
import torch.nn.functional as F


@MODELS.register_module()
class PoseSegUnet(BaseModule):
    def __init__(self, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential()

        self.fuse_up32 = nn.Sequential(
            nn.Conv2d(2048, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

        self.fuse_up16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer0, self.deconv_layer1, self.deconv_layer2 = self._make_deconv_layer()

        self.num_joints = 33
        decoder_channels = 64
        self.final_coarse_heatmap_layer = nn.Sequential(
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
        self.final_refine_heatmap_layer = nn.Sequential(
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

        self.final_refine_offset_layer = nn.Sequential(
            nn.Conv2d(decoder_channels+2, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.num_joints * 2, kernel_size=3, stride=1, padding=1
            )
        )

        self.final_coarse_seg_layer = nn.Sequential(
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

        self.final_refine_seg_layer = nn.Sequential(
            nn.Conv2d(decoder_channels+self.num_joints, decoder_channels, kernel_size=3, stride=1, padding=1),
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
        self.criterion = torch.nn.MSELoss(reduction='none' if self.ohem else 'mean')

        self.seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=30.0,
                     reduction='mean'
                )
            )

        self.offset_loss_weight = 1.0
        self.heatmap_loss_weight = 40.0
        if train_cfg is not None:
            self.offset_loss_weight = train_cfg.get('offset_loss_weight', 0.1)
            self.heatmap_loss_weight = train_cfg.get('heatmap_loss_weight', 1.0)

    def _make_deconv_layer(self):
        deconv_layer0 = []
        deconv_layer1 = []
        deconv_layer2 = []
        deconv1 = nn.Upsample(scale_factor=2, mode="nearest")  # ,align_corners=False
        conv1_1x1 = nn.Conv2d(
            160, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)

        deconv2 = nn.Upsample(scale_factor=2, mode="nearest")
        conv2_1x1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(64)

        deconv3 = nn.Upsample(scale_factor=2, mode="nearest")
        conv3_1x1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(64)

        deconv_layer0.append(deconv1)
        deconv_layer0.append(conv1_1x1)
        deconv_layer0.append(bn1)
        deconv_layer0.append(nn.ReLU(inplace=True))

        deconv_layer1.append(deconv2)
        deconv_layer1.append(conv2_1x1)
        deconv_layer1.append(bn2)
        deconv_layer1.append(nn.ReLU(inplace=True))

        deconv_layer2.append(deconv3)
        deconv_layer2.append(conv3_1x1)
        deconv_layer2.append(bn3)
        deconv_layer2.append(nn.ReLU(inplace=True))

        return [nn.Sequential(*deconv_layer0), nn.Sequential(*deconv_layer1), nn.Sequential(*deconv_layer2)]

    def forward_train(self, image, heatmap, offset_x, offset_y, heatmap_weight, joints_vis, segments, **kwargs):
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x4 = self.model.layer1(x)
        x8 = self.model.layer2(x4)
        x16 = self.model.layer3(x8)
        x32 = self.model.layer4(x16)

        x32_fuse = self.fuse_up32(x32)
        up_16 = self.deconv_layer0(x32_fuse)
        up_16_fuse = self.fuse_up16(up_16)        
        up_8 = self.deconv_layer1(up_16_fuse)
        output_layout = self.deconv_layer2(up_8)

        #--------------------- coarse predict seg/pose--------------------#
        layout_seg_coarse_logits = self.final_coarse_seg_layer(output_layout)
        layout_pose = torch.concat([output_layout, torch.softmax(layout_seg_coarse_logits, 1)], 1)

        layout_uv_coarse_heatmap = self.final_coarse_heatmap_layer(output_layout)
        layout_seg = torch.concat([output_layout, torch.sigmoid(layout_uv_coarse_heatmap)], 1)

        #--------------------- refine predict seg/pose---------------------#
        layout_seg_refine_logits = self.final_refine_seg_layer(layout_seg)
        layout_uv_refine_heatmap = self.final_refine_heatmap_layer(layout_pose)
        layout_uv_refine_off = self.final_refine_offset_layer(layout_pose)
 
        #----------------------- compuate loss ----------------------------#
        heatmap = heatmap[kwargs['has_joints']]
        heatmap_weight = heatmap_weight[kwargs['has_joints']]
        offset_x = offset_x[kwargs['has_joints']]
        offset_y = offset_y[kwargs['has_joints']]
        joints_vis = joints_vis[kwargs['has_joints']]

        seg_labels = segments.to(torch.int64)
        seg_labels = seg_labels[kwargs['has_segments']]

        # part1 coarse	
        layout_uv_coarse_heatmap = layout_uv_coarse_heatmap[kwargs['has_joints']]
        layout_seg_coarse_logits = layout_seg_coarse_logits[kwargs['has_segments']]
        layout_pose_coarse_loss_output = \
            self._compute_loss_with_heatmap(
                layout_uv_coarse_heatmap, None, heatmap, heatmap_weight, offset_x, offset_y, joints_vis)

        layout_seg_coarse_logits = F.interpolate(layout_seg_coarse_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_seg_coarse_loss_value = self.seg_loss(layout_seg_coarse_logits, seg_labels, ignore_index=255)

        # part2 refine
        layout_uv_refine_heatmap = layout_uv_refine_heatmap[kwargs['has_joints']]
        layout_uv_refine_off = layout_uv_refine_off[kwargs['has_joints']]
        layout_seg_refine_logits = layout_seg_refine_logits[kwargs['has_segments']]
        layout_pose_refine_loss_output = \
            self._compute_loss_with_heatmap(
                layout_uv_refine_heatmap, layout_uv_refine_off, heatmap, heatmap_weight, offset_x, offset_y, joints_vis)

        layout_seg_refine_logits = F.interpolate(layout_seg_refine_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_seg_refine_loss_value = self.seg_loss(layout_seg_refine_logits, seg_labels, ignore_index=255)

        loss_output = {
            'layout_coarse_loss_uv_hm': layout_pose_coarse_loss_output['loss_uv_hm'],
            'layout_coarse_loss_seg': layout_seg_coarse_loss_value,          
            'layout_refine_loss_uv_hm': layout_pose_refine_loss_output['loss_uv_hm'],
            'layout_refine_loss_xy_offset': layout_pose_refine_loss_output['loss_xy_offset'],
            'layout_refine_loss_seg': layout_seg_refine_loss_value,          
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
        image_h,image_w = image.shape[:2]
        x32, x16, x8, x4 = self.backbone(image)   # x32,x16,x8,x4
        x4_fuse = self.fuse_up32(x4)
        up8 = self.deconv_layer0(x4_fuse)
        up16 = self.deconv_layer1(up8)
        up16_fuse = self.fuse_up16(up16)
        output = self.deconv_layer2(up16_fuse)

        seg_logits = self.final_seg_layer_1(output)
        seg_pred = torch.softmax(seg_logits, 1)
        # output = torch.concat([output, seg_pred], 1)

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

        results = {
            'pred_joints2d': [sample_joint2d for sample_joint2d in pred_gt_hm],
            'pred_seg': seg_pred.detach().cpu().numpy()
        }
        return results

    def _loss(self, pred_heatmap, pred_offset_xy, gt_heatmap, joint_mask, heatmap_mask, offset_x, offset_y):
        loss_hm = self.cls_criterion(pred_heatmap, gt_heatmap)
        joint_num = pred_heatmap.shape[1]
        loss_hm = loss_hm * joint_mask

        hard_weight = 40
        mid_weight = 20
        easy_weight = 10
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
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x4 = self.model.layer1(x)
        x8 = self.model.layer2(x4)
        x16 = self.model.layer3(x8)
        x32 = self.model.layer4(x16)

        x32_fuse = self.fuse_up32(x32)
        up_16 = self.deconv_layer0(x32_fuse)
        up_16_fuse = self.fuse_up16(up_16)        
        up_8 = self.deconv_layer1(up_16_fuse)
        output_layout = self.deconv_layer2(up_8)        

        #--------------------- coarse predict seg/pose--------------------#
        layout_seg_coarse_logits = self.final_coarse_seg_layer(output_layout)
        layout_pose = torch.concat([output_layout, torch.softmax(layout_seg_coarse_logits, 1)], 1)

        layout_uv_coarse_heatmap = self.final_coarse_heatmap_layer(output_layout)
        layout_seg = torch.concat([output_layout, torch.sigmoid(layout_uv_coarse_heatmap)], 1)

        #--------------------- refine predict seg/pose---------------------#
        layout_seg_refine_logits = self.final_refine_seg_layer(layout_seg)
        layout_uv_refine_heatmap = self.final_refine_heatmap_layer(layout_pose)
        layout_uv_refine_off = self.final_refine_offset_layer(layout_pose)
 
        seg_pred = torch.softmax(layout_seg_refine_logits, 1)
        layout_uv_refine_heatmap = torch.sigmoid(layout_uv_refine_heatmap)
        layout_uv_refine_heatmap = F.max_pool2d(layout_uv_refine_heatmap, 3, stride=1, padding=(3 - 1) // 2)
        return layout_uv_refine_heatmap, layout_uv_refine_off, seg_pred
