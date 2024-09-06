import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn import ConvModule
from antgo.framework.helper.runner import BaseModule, ModuleList
from models.utils.reparam_layers import RepVGGBlock
from models.utils.repvgg import *


class CrossM(nn.Module):
    def __init__(self, mid_channels=64):
        super(CrossM, self).__init__()
        self.cross_l = ConvModule(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1, 
                    padding=2, 
                    dilation=2,
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))
        self.proj_l = \
            RepVGGBlock(
                in_channels=mid_channels, 
                out_channels=mid_channels, 
                stride=1, 
                padding=1,
                without_branch_norm=False
            )

    def forward(self, high_x1, low_x2):
        h = F.relu(self.cross_l(high_x1) + low_x2)
        h = self.proj_l(h)
        return h


class DeltaM(nn.Module):
    def __init__(self, mid_channels=64):
        super(DeltaM, self).__init__()
        self.delta_l = ConvModule(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1, 
                    padding=2,
                    dilation=2,
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))

        self.proj_l = \
            RepVGGBlock(
                in_channels=mid_channels, 
                out_channels=mid_channels, 
                stride=1, 
                padding=1,
                without_branch_norm=False
            )

    def forward(self, x1, x2):
        y = F.relu(x1 + self.delta_l(x2))
        z = self.proj_l(y)
        return z


class CSPRepLayer(nn.Module):
    """CSPRepLayer, a layer that combines Cross Stage Partial Networks with
    RepVGG Blocks.

    Args:
        in_channels (int): Number of input channels to the layer.
        out_channels (int): Number of output channels from the layer.
        num_blocks (int): The number of RepVGG blocks to be used in the layer.
            Defaults to 3.
        widen_factor (float): Expansion factor for intermediate channels.
            Determines the hidden channel size based on out_channels.
            Defaults to 1.0.
        norm_cfg (dict): Configuration for normalization layers.
            Defaults to Batch Normalization with trainable parameters.
        act_cfg (dict): Configuration for activation layers.
            Defaults to SiLU (Swish) with in-place operation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 widen_factor: float = 1.0,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * widen_factor)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class APPM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(APPM, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # level 0
        self.conv_l0 = ConvModule(
            self.in_channels,
            self.mid_channels,
            1,
            dilation=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        # level 1
        self.conv_l1_0 = ConvModule(
            self.in_channels,
            self.mid_channels,
            3,
            dilation=2,
            padding=2,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')            
        )
        self.conv_l1_1 = ConvModule(
            self.mid_channels,
            self.mid_channels,
            3,
            dilation=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')            
        )

        # level 2
        self.conv_l2_0 = ConvModule(
            self.in_channels,
            self.mid_channels,
            3,
            dilation=4,
            padding=4,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.conv_l2_1 = ConvModule(
            self.mid_channels,
            self.mid_channels,
            3,
            dilation=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')            
        )

        # project
        self.project_l = ConvModule(
            self.mid_channels*3,
            self.out_channels,
            1,
            dilation=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        out_0 = self.conv_l0(x)
        
        out_1_pre = self.conv_l1_0(x)
        out_1 = self.conv_l1_1(out_0+out_1_pre)

        out_2_pre = self.conv_l2_0(x)
        out_2 = self.conv_l2_1(out_1+out_2_pre)

        out = torch.cat([out_0, out_1, out_2], 1)
        out = self.project_l(out)
        return out


@BACKBONES.register_module()
class UDMRepVGG(nn.Module):
    def __init__(self,arc, pretrained=False, in_channels=3, mid_channels=64, out_channels=64, delta_mode=True, pan_mode=True, deepen_factor=1.0, widen_factor=1.0, out_strides=[16,32]):
        super(UDMRepVGG, self).__init__()
        self.arc = arc

        self.delta_mode = delta_mode
        self.pan_mode = pan_mode
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor

        backbone_model_map = {
            'RepVGG-A0': create_RepVGG_A0,
            'RepVGG-A1': create_RepVGG_A1,
            'RepVGG-A2': create_RepVGG_A2,
            'RepVGG-B0': create_RepVGG_B0,
            'RepVGG-B1': create_RepVGG_B1,
            'RepVGG-B1g2': create_RepVGG_B1g2,
            'RepVGG-B1g4': create_RepVGG_B1g4,
            'RepVGG-B2': create_RepVGG_B2,
            'RepVGG-B2g2': create_RepVGG_B2g2,
            'RepVGG-B2g4': create_RepVGG_B2g4,
            'RepVGG-B3': create_RepVGG_B3,
            'RepVGG-B3g2': create_RepVGG_B3g2,
            'RepVGG-B3g4': create_RepVGG_B3g4,
            'RepVGG-D2se': create_RepVGG_D2se
        }

        backbone_channels_map = {
            'RepVGG-A0': [48,96,192,1280],
            'RepVGG-A1': [64,128,256,1280],
            'RepVGG-B0': [64,128,256,1280],
            'RepVGG-B1': [128,256,512,2048],
            'RepVGG-B1g4': [128,256,512,2048]
        }
        backbone = backbone_model_map[self.arc]()

        if pretrained:
            checkpoint = torch.load(pretrained)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            # replace key name
            renamed_checkpoint = {}
            for k, v in checkpoint.items():
                if 'rbr_dense' in k:
                    k = k.replace('rbr_dense', 'branch_3x3')
                if 'rbr_1x1' in k:
                    k = k.replace('rbr_1x1', 'branch_1x1')
                if 'rbr_identity' in k:
                    k = k.replace('rbr_identity', 'branch_norm')
                renamed_checkpoint[k] = v
            backbone.load_state_dict(renamed_checkpoint)

        self.backbone = backbone
        self.backbone.linear = None
        self.backbone.gap = None
        self.backbone_c_level_0 = backbone_channels_map[self.arc][0]
        self.backbone_c_level_1 = backbone_channels_map[self.arc][1]
        self.backbone_c_level_2 = backbone_channels_map[self.arc][2]
        self.backbone_c_level_3 = backbone_channels_map[self.arc][3]

        # 1/4 ,  1/8, 1/16, 1/32
        # (dilation(k-1)+1-stride)/2
        self.down4_to_16 = \
                ConvModule(
                    mid_channels, 
                    self.backbone_c_level_2,
                    kernel_size=5,
                    stride=4, 
                    padding=(int)((1*(5-1)+1-4)//2+1), 
                    dilation=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    mid_channels, 
                    self.backbone_c_level_3,
                    kernel_size=5,
                    stride=8,
                    padding=2, 
                    dilation=2, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))

        self.compression8_to_4 = \
                ConvModule(
                    self.backbone_c_level_1, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    self.backbone_c_level_2, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.compression32_to_4 = \
                ConvModule(
                    self.backbone_c_level_3, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))

        self.cross_0 = RepVGGBlock(
            in_channels=self.backbone_c_level_0, 
            out_channels=mid_channels, 
            stride=1, 
            padding=1
        )
        self.cross_1 = CrossM(mid_channels)
        self.cross_2 = CrossM(mid_channels)
        self.cross_3 = CrossM(mid_channels)

        # appm
        self.appm = APPM(in_channels=self.backbone_c_level_3, mid_channels=420, out_channels=self.backbone_c_level_3)

        # export stries level
        self.out_strides = out_strides

        if self.delta_mode:
            self.delta_conv_1 = DeltaM(mid_channels)
            self.delta_conv_2 = DeltaM(mid_channels)

        if self.pan_mode:
            downsample_convs = list()
            pan_blocks = list()
            lateral_convs = list()

            # 8,16,32
            lateral_channels = [self.backbone_c_level_1, self.backbone_c_level_2, self.backbone_c_level_3]
            for idx in range(3):
                lateral_convs.append(
                    ConvModule(
                        lateral_channels[idx],
                        mid_channels,
                        1,
                        stride=1,
                        padding=0,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU'))
                )
                downsample_convs.append(
                    ConvModule(
                        mid_channels if idx == 0 else out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU')))
                pan_blocks.append(
                    CSPRepLayer(
                        mid_channels + out_channels,
                        out_channels,
                        round(3 * deepen_factor),
                        act_cfg=dict(type='ReLU'),
                        widen_factor=widen_factor))

            self.lateral_convs = ModuleList(lateral_convs)
            self.downsample_convs = ModuleList(downsample_convs)
            self.pan_blocks = ModuleList(pan_blocks)

    def forward(self, x):
        x = self.backbone.stage0(x)             # 1/2
        x4 = self.backbone.stage1(x)            # 1/4

        out_layer_1 = self.cross_0(x4)
        x8 = self.backbone.stage2(x4)           # 1/8

        compression8_to_4 = self.compression8_to_4(x8)
        out_layer_2 = \
            self.cross_1(
                out_layer_1,
                F.interpolate(
                    compression8_to_4, scale_factor=2, mode='bilinear'
                )
            )

        x16 = self.backbone.stage3[0](x8)
        x16 = F.relu(x16 + self.down4_to_16(out_layer_2))
        x16 = self.backbone.stage3[1:](x16)

        compression16_to_4 = self.compression16_to_4(x16)
        out_layer_3 = \
            self.cross_2(
                out_layer_2,
                F.interpolate(
                    compression16_to_4, scale_factor=4, mode='bilinear'
                )
            )

        x32 = self.backbone.stage4[0](x16)
        x32 = F.relu(x32 + self.down4_to_32(out_layer_3))
        x32 = self.backbone.stage4[1:](x32)

        # appm
        x32 = self.appm(x32)
        compression32_to_4 = self.compression32_to_4(x32)

        out_layer_4 = \
            self.cross_3(
                out_layer_3,
                F.interpolate(
                    compression32_to_4, scale_factor=8, mode='bilinear'
                )
            )

        out_list = []
        if self.delta_mode:
            # base
            out_list.append(out_layer_4)

            # fix with delta
            out_layer_4 = self.delta_conv_1(out_list[-1], out_layer_3)
            out_list.append(out_layer_4)

            # fix with delta
            out_layer_4 = self.delta_conv_2(out_list[-1], out_layer_2)
            out_list.append(out_layer_4)

        if self.pan_mode:
            out_list = []
            # 1/4
            out = out_layer_4
            out_list.append(out)

            # 1/8
            downsample_feat = self.downsample_convs[0](out_list[-1])  # Conv
            out = self.pan_blocks[0](
                torch.cat([downsample_feat, self.lateral_convs[0](x8)], axis=1))
            out_list.append(out)

            # 1/16
            downsample_feat = self.downsample_convs[1](out_list[-1])
            out = self.pan_blocks[1](
                torch.cat([downsample_feat, self.lateral_convs[1](x16)], axis=1)
            )
            out_list.append(out)

            # 1/32
            downsample_feat = self.downsample_convs[2](out_list[-1])
            out = self.pan_blocks[2](
                torch.cat([downsample_feat, self.lateral_convs[2](x32)], axis=1)
            )
            out_list.append(out)

            # filter out list
            index_map = {4: 0, 8: 1, 16: 2, 32: 3}
            filter_out_list = [out_list[index_map[s]] for s in self.out_strides]
            return filter_out_list

        return out_list
