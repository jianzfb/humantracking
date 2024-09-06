from itertools import zip_longest
from typing import List, Optional, Union

from antgo.framework.helper.utils import is_list_of
from torch import Tensor
import torch

from .base import BasePoseEstimator
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss
from antgo.framework.helper.runner import BaseModule
from models.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from antgo.framework.helper.structures import InstanceData, PixelData
from models.structures.pose_data_sample import PoseDataSample
from models.structures.bbox.bbox_overlaps import bbox_overlaps
import numpy as np
import cv2
import os


@MODELS.register_module()
class BottomupPoseEstimator(BasePoseEstimator):
    """Base class for bottom-up pose estimators.

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
            ``None``.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def forward_test(self, image, bbox, keypoints, keypoints_visible, **kwargs):
        feats = self.extract_feat(image)

        sample_list = []
        for bbox_in_sample, joint_in_sample, joint_vis_in_sample, bbox_label_in_sample, bbox_area_in_sample in zip(bbox, keypoints, keypoints_visible, kwargs['bbox_labels'], kwargs['area']):
            gt_instances = InstanceData()
            gt_instances.bboxes = bbox_in_sample
            gt_instances.labels = bbox_label_in_sample
            gt_instances.keypoints = joint_in_sample
            gt_instances.keypoints_visible = joint_vis_in_sample
            gt_instances.areas = bbox_area_in_sample

            data_sample = PoseDataSample(gt_instances=gt_instances)
            sample_list.append(data_sample)

        # 预测
        preds = self.head.predict(feats, sample_list, test_cfg=self.test_cfg)

        # 转换到原始图像空间坐标
        pred_joints2d = []
        pred_joints_vis = []
        pred_bbox = []
        pred_bbox_score = []
        pred_joints_score = []
        img_id = []
        for single_gt_bboxes, single_pred_info, single_img_id, single_inv_warp_mat in zip(bbox, preds, kwargs['img_id'], kwargs['inv_warp_mat']):
            if len(single_pred_info.bboxes) == 0:
                pred_joints2d.append(np.empty((0,33,2)))
                pred_joints_vis.append(np.empty((0,33)))
                pred_bbox.append(np.empty((0,4)))
                pred_bbox_score.append(np.empty((0)))
                pred_joints_score.append(np.empty((0,33)))
                img_id.append(single_img_id)
                continue

            bbox_ious = bbox_overlaps(single_gt_bboxes, torch.from_numpy(single_pred_info.bboxes).to(single_gt_bboxes.device))
            bbox_ious = bbox_ious.detach().cpu().numpy()
            select_i = np.argmax(bbox_ious, -1)

            single_pred_keypoints = []
            single_pred_keypoint_vis = []
            single_pred_keypoint_score = []
            single_pred_bbox = []
            single_pred_bbox_score = []
            for ii in select_i:
                single_pred_keypoints.append(single_pred_info.keypoints[ii])
                single_pred_keypoint_vis.append(single_pred_info.keypoints_visible[ii])
                single_pred_bbox.append(single_pred_info.bboxes[ii])
                single_pred_bbox_score.append(single_pred_info.bbox_scores[ii])
                single_pred_keypoint_score.append(single_pred_info.keypoints_score[ii])

            single_pred_keypoints = np.stack(single_pred_keypoints, 0)
            single_pred_keypoints = cv2.transform(single_pred_keypoints, single_inv_warp_mat.cpu().numpy())

            single_pred_keypoint_vis = np.stack(single_pred_keypoint_vis, 0)
            single_pred_bbox = np.stack(single_pred_bbox, 0)
            single_pred_bbox_reshape = single_pred_bbox.reshape(1, -1, 2)
            single_pred_bbox_reshape = cv2.transform(single_pred_bbox_reshape, single_inv_warp_mat.cpu().numpy())
            single_pred_bbox = single_pred_bbox_reshape.reshape(-1, 4)

            single_pred_bbox_score = np.stack(single_pred_bbox_score, 0)
            single_pred_keypoint_score = np.stack(single_pred_keypoint_score, 0)

            pred_joints2d.append(single_pred_keypoints)
            pred_joints_vis.append(single_pred_keypoint_vis)
            pred_bbox.append(single_pred_bbox)
            pred_bbox_score.append(single_pred_bbox_score)
            pred_joints_score.append(single_pred_keypoint_score)
            img_id.append(single_img_id)

        results = {
            'pred_keypoint_scores': pred_joints_score,
            'pred_keypoints': pred_joints2d, 
            'pred_bbox_scores': pred_bbox_score,
            'pred_bbox': pred_bbox,
            'pred_img_id': img_id
        }
        return results

    def onnx_export(self, image):
        feats = self.extract_feat(image)
        cls_scores, bbox_preds, kpt_offset, kpt_vis, pose_vecs = self.head.forward(feats)

        # 目标可见性
        flatten_cls_scores = self.head._flatten_predictions(cls_scores).sigmoid()
        # 目标框位置
        flatten_bbox_preds = self.head._flatten_predictions(bbox_preds)
        # 关键点可见性
        flatten_kpt_vis = self.head._flatten_predictions(kpt_vis).sigmoid()
        # 关键点偏移
        flatten_kpt_offset = self.head._flatten_predictions(kpt_offset)
        return flatten_cls_scores, flatten_bbox_preds, flatten_kpt_vis, flatten_kpt_offset
