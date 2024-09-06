# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
# from mmengine.dist import get_dist_info
# from mmengine.structures import PixelData
from torch import Tensor, nn

from models.codecs.utils import get_simcc_normalized
from antgo.framework.helper.runner import BaseModule
from models.losses.classification_loss import *
from models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck, build_loss


OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCHead(BaseModule):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        in_featuremap_size,
        simcc_split_ratio = 2.0,
        final_layer_kernel_size = 1,
        gau_cfg = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        decoder = None,
        init_cfg = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = KLDiscretLoss(
            use_target_weight=True,
            beta=10.,
            label_softmax=True)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)

        # self.mlp = nn.Sequential(
        #     ScaleNorm(flatten_dims),
        #     nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        # TODO
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))


        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        # self.gau = RTMCCBlock(
        #     self.out_channels,
        #     gau_cfg['hidden_dims'],
        #     gau_cfg['hidden_dims'],
        #     s=gau_cfg['s'],
        #     expansion_factor=gau_cfg['expansion_factor'],
        #     dropout_rate=gau_cfg['dropout_rate'],
        #     drop_path=gau_cfg['drop_path'],
        #     attn_type='self-attn',
        #     act_fn=gau_cfg['act_fn'],
        #     use_rel_bias=gau_cfg['use_rel_bias'],
        #     pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)  # -> B, K, hidden

        # # TODO, 取消gau模块
        # feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    def loss(
        self,
        feats,
        keypoint_x_labels, keypoint_y_labels, keypoint_weights,
        train_cfg,
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        # gt_x = torch.cat([
        #     d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        # ],
        #                  dim=0)
        gt_x = keypoint_x_labels
        # gt_y = torch.cat([
        #     d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        # ],
        #                  dim=0)
        gt_y = keypoint_y_labels
        # keypoint_weights = torch.cat(
        #     [
        #         d.gt_instance_labels.keypoint_weights
        #         for d in batch_data_samples
        #     ],
        #     dim=0,
        # )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)
        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
