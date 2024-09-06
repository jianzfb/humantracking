# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional

from torch import Tensor
import torchvision
from torch import nn
import torch.nn.functional as F
# from mmpose.registry import MODELS
# from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
#                                  OptMultiConfig, PixelDataList, SampleList)
# from .base import BasePoseEstimator
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule


@MODELS.register_module()
class TopdownPoseEstimator(BaseModule):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone=None,
                 neck = None,
                 head = None,
                 train_cfg = None,
                 test_cfg = None,
                 data_preprocessor = None,
                 init_cfg = None):
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.backbone = torchvision.models.resnet34(pretrained=True)
        self.backbone.fc = nn.Sequential()
        self.with_neck = False
        if neck is not None:
            self.neck = MODELS.build(neck)
            self.with_neck = True

        self.with_head = False
        if head is not None:
            self.head = MODELS.build(head)
            self.with_head = True

    def forward_train(self, image, keypoint_x_labels, keypoint_y_labels, keypoint_weights, **kwargs):
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = (x,)
        if self.with_neck:
            x = self.neck(x)

        losses = dict()
        if self.with_head:
            losses.update(
                self.head.loss(x, keypoint_x_labels, keypoint_y_labels, keypoint_weights, train_cfg=self.train_cfg))

        return losses

    def onnx_export(self, image):
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = (x,)
        if self.with_neck:
            x = self.neck(x)

        if self.with_head:
            x = self.head(x)
        
        x = (F.softmax(x[0]*10, dim=-1), F.softmax(x[1]*10, dim=-1))
        return x
