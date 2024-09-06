from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F
import cv2


@MODELS.register_module()
class ParsingDDRNetV2(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.cls_num = 19
        decoder_channels = 64

        self.parsing_layer_1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.cls_num, kernel_size=1, stride=1, padding=0
            )
        )

        self.parsing_layer_2 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),            
            nn.Conv2d(
                decoder_channels, self.cls_num, kernel_size=1, stride=1, padding=0
            )
        )
        class_weight = [1]*19
        class_weight[9] = 5
        class_weight[10] = 5
        self.seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=10.0,
                     reduction='mean',
                     class_weight=class_weight
                )
            )

    def forward_train(self, image, segments, **kwargs):
        output_list = self.backbone(image)
        output_layout_1, output_layout_2 = output_list[:2]

        seg_labels = segments.to(torch.int64)

        #---------------------layout_1 seg loss---------------------------#
        layout_1_seg_logits = self.parsing_layer_1(output_layout_1)
        layout_1_seg_logits = F.interpolate(layout_1_seg_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_1_seg_loss_value = self.seg_loss(layout_1_seg_logits, seg_labels, ignore_index=255)

        #---------------------layout_2 seg loss---------------------------#
        layout_2_seg_logits = self.parsing_layer_2(output_layout_2)
        layout_2_seg_logits = F.interpolate(layout_2_seg_logits, seg_labels.shape[1:], mode='bilinear', align_corners=True)
        layout_2_seg_loss_value = self.seg_loss(layout_2_seg_logits, seg_labels, ignore_index=255)

        loss_output = {
            'layout_1_loss_seg': layout_1_seg_loss_value,
            'layout_2_loss_seg': layout_2_seg_loss_value * 0.4,            
        }
        return loss_output

    def forward_test(self, image, **kwargs):
        image_h, image_w = image.shape[2:]
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, output_layout_2 = output_list[:2]

        layout_1_seg_logits = self.parsing_layer_1(output_layout_1)
        layout_1_seg_logits = F.interpolate(layout_1_seg_logits, (image_h, image_w), mode='bilinear', align_corners=True)
        seg_pred = torch.softmax(layout_1_seg_logits, 1)

        results = {
            'pred_seg': seg_pred.detach().cpu().numpy()
        }
        return results

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout_1, _ = output_list[:2]

        layout_1_seg_logits = self.parsing_layer_1(output_layout_1)
        seg_pred = torch.softmax(layout_1_seg_logits, 1)

        return seg_pred