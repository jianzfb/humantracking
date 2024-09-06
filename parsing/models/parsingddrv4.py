from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F
import cv2


@MODELS.register_module()
class ParsingDDRNetV4(BaseModule):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.cls_num = 19
        decoder_channels = 64

        self.parsing_layer = nn.Sequential(
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

        self.coarse_body_layer = nn.Sequential(
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
        class_weight = [1]*19
        class_weight[9] = 5
        class_weight[10] = 5
        self.parsing_seg_loss = \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=10.0,
                     reduction='mean',
                     class_weight=class_weight
                )
            )
        self.body_seg_loss =             \
            build_loss(
                dict(
                     type='CrossEntropyLoss',
                     loss_weight=5.0,
                     use_sigmoid=True,
                     reduction='mean'
                )
            )

    def forward_train(self, image, whole_body_segment, parsing_body_segment, **kwargs):
        output_list = self.backbone(image)
        output_layout, _ = output_list[:2]

        whole_body_segment = whole_body_segment.to(torch.int64)
        parsing_body_segment = parsing_body_segment.to(torch.int64)

        whole_body_logits = self.coarse_body_layer(output_layout)
        resized_whole_body_logits = F.interpolate(whole_body_logits, whole_body_segment.shape[1:], mode='bilinear', align_corners=True)
        whole_body_loss = self.body_seg_loss(resized_whole_body_logits, whole_body_segment, ignore_index=255)

        whole_body_sigmoid = torch.sigmoid(whole_body_logits)
        output_layout = output_layout * whole_body_sigmoid
        parsing_body_logits = self.parsing_layer(output_layout)
        # parsing_body_pred = torch.softmax(parsing_body_logits, 1)
        # refine_whole_body_logits = whole_body_logits * parsing_body_pred
        # refine_whole_body_logits = self.refine_body_layer(refine_whole_body_logits)
        # refine_resized_whole_body_logits = F.interpolate(refine_whole_body_logits, whole_body_segment.shape[1:], mode='bilinear', align_corners=True)
        # refine_whole_body_loss = self.body_seg_loss(refine_resized_whole_body_logits, whole_body_segment, ignore_index=255)

        has_parsing_body = kwargs['has_parsing_body']
        parsing_body_segment = parsing_body_segment[has_parsing_body]   
        parsing_body_logits = parsing_body_logits[has_parsing_body]

        parsing_body_logits = F.interpolate(parsing_body_logits, parsing_body_segment.shape[1:], mode='bilinear', align_corners=True)
        parsing_body_loss = self.parsing_seg_loss(parsing_body_logits, parsing_body_segment, ignore_index=255)

        loss_output = {
            'parsing_body_loss': parsing_body_loss,
            'whole_body_loss': whole_body_loss,
        }
        return loss_output

    def forward_test(self, image, **kwargs):
        image_h, image_w = image.shape[2:]
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout, _ = output_list[:2]

        whole_body_logits = self.coarse_body_layer(output_layout)
        whole_body_sigmoid = torch.sigmoid(whole_body_logits)
        output_layout = output_layout * whole_body_sigmoid
        parsing_body_logits = self.parsing_layer(output_layout)

        parsing_body_logits = F.interpolate(parsing_body_logits, (image_h, image_w), mode='bilinear', align_corners=True)
        seg_pred = torch.softmax(parsing_body_logits, 1)
        seg_pred = torch.argmax(seg_pred, dim=1)

        results = {
            'pred_segments': seg_pred.detach().cpu().numpy()
        }
        return results

    def onnx_export(self, image):
        output_list = self.backbone(image)   # x32,x16,x8,x4
        output_layout, _ = output_list[:2]

        whole_body_logits = self.coarse_body_layer(output_layout)
        whole_body_sigmoid = torch.sigmoid(whole_body_logits)
        output_layout = output_layout * whole_body_sigmoid
        parsing_body_logits = self.parsing_layer(output_layout)
        seg_pred = torch.softmax(parsing_body_logits, 1)

        return seg_pred