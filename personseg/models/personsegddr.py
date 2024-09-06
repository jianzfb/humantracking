from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F
import cv2


@MODELS.register_module()
class PersonSegDDR(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        decoder_channels = 64

        self.final_seg_layer = nn.Sequential(
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

        self.seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=5.0,
                     reduction='mean'
                )
            )

    def forward_train(self, image, segments, **kwargs):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, output_layout_2 = output_list[:2]

        seg_labels = segments.to(torch.int64)
        layout_1_seg_logits = self.final_seg_layer(output_layout_1)
        layout_1_seg_logits = F.interpolate(layout_1_seg_logits, seg_labels.shape[2:], mode='bilinear', align_corners=True)
        layout_1_seg_loss_value = self.seg_loss(layout_1_seg_logits.squeeze(1), seg_labels, ignore_index=255)

        loss_output = {
            'layout_1_seg_loss': layout_1_seg_loss_value
        }
        return loss_output

    def forward_test(self, image, **kwargs):
        image_h, image_w = image.shape[2:]
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, output_layout_2 = output_list[:2]
        output = output_layout_1

        seg_logits = self.final_seg_layer(output)
        seg_pred = torch.sigmoid(seg_logits)

        results = {
            'pred_seg': seg_pred.detach().cpu().numpy()
        }
        return results

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output = output_list[0]

        seg_logits = self.final_seg_layer(output)
        seg_pred = torch.sigmoid(seg_logits)

        return seg_pred
