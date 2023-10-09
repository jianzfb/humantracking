import enum
from antgo.framework.helper.models.builder import DETECTORS
from antgo.framework.helper.models.detectors.single_stage import SingleStageDetector, to_image_list
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms 


@DETECTORS.register_module()
class YoloX(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YoloX, self).__init__(backbone, neck, bbox_head,
                                     train_cfg, test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      image,
                      image_meta,                      
                      bboxes,
                      labels,
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
            bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            labels (list[Tensor]): Class indices corresponding to each box
            bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        image_list, image_meta = to_image_list(image, image_meta)
        image = image_list.tensors
        super(SingleStageDetector, self).forward_train(image, image_meta)
        x = self.extract_feat(image)

        batch_size = image.shape[0]
        label_bboxes = []
        for b_i in range(batch_size):
            if len(labels[b_i].shape) == 1:
                labels[b_i] = labels[b_i].view(-1, 1)

            cxcywh_bboxes = bboxes[b_i]
            if cxcywh_bboxes.shape[0] > 0:
                # xxyy -> cxcywh
                x0 = cxcywh_bboxes[:,0:1]
                y0 = cxcywh_bboxes[:,1:2]
                x1 = cxcywh_bboxes[:,2:3]
                y1 = cxcywh_bboxes[:,3:4]

                width = x1-x0
                height = y1-y0
                cx = x0+width/2.0
                cy = y0+height/2.0

                cxcywh_bboxes = torch.cat([cx,cy,width,height], -1)
            label_bboxes.append(
                torch.cat([labels[b_i], cxcywh_bboxes], -1)
            )
        losses = self.bbox_head(x, label_bboxes, image)
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
        for b_i in range(batch_size):
            obj_mask = obj_score[b_i] > 0.2
            select_box = xyxy[b_i][obj_mask]
            select_prob_dis = cls_score[b_i][obj_mask]
            select_label = torch.argmax(select_prob_dis, dim=-1)
            select_prob_mask = torch.zeros_like(select_prob_dis,dtype=torch.bool).scatter_(-1, select_label.unsqueeze(-1), 1)
            select_prob = select_prob_dis[select_prob_mask]
            keep = batched_nms(
                select_box,
                select_prob,
                select_label, 
                0.1)

            box_list.append(torch.cat([select_box[keep], select_prob[keep].unsqueeze(-1)], -1))
            label_list.append(select_label[keep])

        bbox_results = {
            'box': box_list,
            'label': label_list,
        }

        return bbox_results

    def onnx_export(self, image):
        feat = self.extract_feat(image)
        self.bbox_head.decode_in_inference = False
        outputs = self.bbox_head(feat)

        return outputs
