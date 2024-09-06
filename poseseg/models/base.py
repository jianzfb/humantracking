# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
from torch import Tensor
from models.structures.pose_data_sample import PoseDataSample
from antgo.framework.helper.structures import InstanceData, PixelData

from models.utils.typing import (ConfigType, ForwardResults, OptConfigType,
                                 Optional, OptMultiConfig, OptSampleList,
                                 SampleList)
import cv2

class BasePoseEstimator(BaseModule, metaclass=ABCMeta):
    """Base class for pose estimators.

    Args:
        data_preprocessor (dict | ConfigDict, optional): The pre-processing
            config of :class:`BaseDataPreprocessor`. Defaults to ``None``
        init_cfg (dict | ConfigDict): The model initialization config.
            Defaults to ``None``
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        # the PR #2108 and #2126 modified the interface of neck and head.
        # The following function automatically detects outdated
        # configurations and updates them accordingly, while also providing
        # clear and concise information on the changes made.
        # neck, head = check_and_update_config(neck, head)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        # Register the hook to automatically convert old version state dicts
        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'head') and self.head is not None

    # def forward(self,
    #             inputs: torch.Tensor,
    #             data_samples: OptSampleList,
    #             mode: str = 'tensor') -> ForwardResults:
    #     """The unified entry for a forward process in both training and test.

    #     The method should accept three modes: 'tensor', 'predict' and 'loss':

    #     - 'tensor': Forward the whole network and return tensor or tuple of
    #     tensor without any post-processing, same as a common nn.Module.
    #     - 'predict': Forward and return the predictions, which are fully
    #     processed to a list of :obj:`PoseDataSample`.
    #     - 'loss': Forward and return a dict of losses according to the given
    #     inputs and data samples.

    #     Note that this method doesn't handle neither back propagation nor
    #     optimizer updating, which are done in the :meth:`train_step`.

    #     Args:
    #         inputs (torch.Tensor): The input tensor with shape
    #             (N, C, ...) in general
    #         data_samples (list[:obj:`PoseDataSample`], optional): The
    #             annotation of every sample. Defaults to ``None``
    #         mode (str): Set the forward mode and return value type. Defaults
    #             to ``'tensor'``

    #     Returns:
    #         The return type depends on ``mode``.

    #         - If ``mode='tensor'``, return a tensor or a tuple of tensors
    #         - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
    #             that contains the pose predictions
    #         - If ``mode='loss'``, return a dict of tensor(s) which is the loss
    #             function value
    #     """
    #     if isinstance(inputs, list):
    #         inputs = torch.stack(inputs)
    #     if mode == 'loss':
    #         return self.loss(inputs, data_samples)
    #     elif mode == 'predict':
    #         # use customed metainfo to override the default metainfo
    #         if self.metainfo is not None:
    #             for data_sample in data_samples:
    #                 data_sample.set_metainfo(self.metainfo)
    #         return self.predict(inputs, data_samples)
    #     elif mode == 'tensor':
    #         return self._forward(inputs)
    #     else:
    #         raise RuntimeError(f'Invalid mode "{mode}". '
    #                            'Only supports loss, predict and tensor mode.')

    def forward_train(self, image, bboxes, joints2d, joints_vis, joints_weight, **kwargs):
        sample_list = []
        for bbox, joint, joint_vis, joint_weight, bbox_label, bbox_area in zip(bboxes, joints2d, joints_vis, joints_weight, kwargs['labels'], kwargs['area']):
            # pose_meta = dict(img_shape=(800, 1216), crop_size=(256, 192), heatmap_size=(64, 48))
            gt_instances = InstanceData()
            gt_instances.bboxes = bbox
            gt_instances.labels = bbox_label
            gt_instances.keypoints = joint
            gt_instances.keypoints_visible = joint_vis
            gt_instances.areas = bbox_area
            gt_instances.keypoints_visible_weights = joint_weight
            # gt_fields = PixelData()
            # gt_fields.heatmaps = torch.rand((33, 64, 48))
            data_sample = PoseDataSample(gt_instances=gt_instances)
            sample_list.append(data_sample)

        return self.loss(image, sample_list)


    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    # @abstractmethod
    # def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
    #     """Predict results from a batch of inputs and data samples with post-
    #     processing."""

    # def _forward(self,
    #              inputs: Tensor,
    #              data_samples: OptSampleList = None
    #              ) -> Union[Tensor, Tuple[Tensor]]:
    #     """Network forward process. Usually includes backbone, neck and head
    #     forward without any post-processing.

    #     Args:
    #         inputs (Tensor): Inputs with shape (N, C, H, W).

    #     Returns:
    #         Union[Tensor | Tuple[Tensor]]: forward output of the network.
    #     """

    #     x = self.extract_feat(inputs)
    #     if self.with_head:
    #         x = self.head.forward(x)

    #     return x

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x
