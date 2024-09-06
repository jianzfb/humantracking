import enum
from antgo.framework.helper.models.builder import DETECTORS
from antgo.framework.helper.models.detectors.single_stage import SingleStageDetector, to_image_list
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms 
import numpy as np
import cv2


@DETECTORS.register_module()
class YoloX(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YoloX, self).__init__(backbone, neck, bbox_head,
                                     train_cfg, test_cfg, init_cfg)

    def forward_train(self,
                      image,
                      image_meta,                      
                      bboxes,
                      bboxes_ignore=None, **kwargs):
        """
        Args:
            image (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            image_meta (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        image_list, image_meta = to_image_list(image, image_meta)
        image = image_list.tensors
        super(SingleStageDetector, self).forward_train(image, image_meta)
        x = self.extract_feat(image)

        if bboxes is None:
            output = self.bbox_head(x, None, image, True)
            return output

        losses = self.bbox_head(x, bboxes, image)
        return losses

    def simple_test(self, image, image_meta, rescale=True, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        image_list, image_meta = to_image_list(image, image_meta)
        image = image_list.tensors

        feat = self.extract_feat(image)
        self.bbox_head.decode_in_inference = True
        results_list = self.bbox_head(feat)

        cxcywh = results_list[:,:,:4]
        obj_score = results_list[:,:,4]
        cls_score = results_list[:,:,5:]

        xy_left_top = cxcywh[:,:,:2] - cxcywh[:,:,2:]/2.0
        xy_right_bottom = cxcywh[:,:,:2] + cxcywh[:,:,2:]/2.0
        xyxy = torch.cat([xy_left_top,xy_right_bottom], -1)

        batch_size = image.shape[0]
        box_list = []
        label_list = []
        # 转换到原始图像分辨率尺寸坐标
        for b_i in range(batch_size):
            obj_mask = obj_score[b_i] > 0.1
            select_box = xyxy[b_i][obj_mask]
            select_prob_dis = cls_score[b_i][obj_mask]
            select_label = torch.argmax(select_prob_dis, dim=-1)
            select_prob_mask = torch.zeros_like(select_prob_dis,dtype=torch.bool).scatter_(-1, select_label.unsqueeze(-1), 1)
            select_prob = select_prob_dis[select_prob_mask]

            select_box = select_box / torch.asarray([image_meta[b_i]['scale_factor']], device=select_box.device) - torch.asarray([image_meta[b_i]['offset']], device=select_box.device)
            keep = batched_nms(
                select_box,
                select_prob,
                select_label, 
                0.2)

            box_list.append(torch.cat([select_box[keep], select_prob[keep].unsqueeze(-1)], -1))
            label_list.append(select_label[keep])

        bbox_results = {
            'pred_bboxes': box_list,
            'pred_labels': label_list,
        }

        return bbox_results

    def onnx_export(self, image):
        feat = self.extract_feat(image)
        self.bbox_head.decode_in_inference = False
        outputs = self.bbox_head(feat)

        return outputs
