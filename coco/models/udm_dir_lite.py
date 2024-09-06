import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn import ConvModule,DepthwiseSeparableConvModule
from antgo.framework.helper.models.segmentation.head.aspp_head import *
from antgo.framework.helper.runner import BaseModule, ModuleList
from models.utils.reparam_layers import RepVGGBlock
from torch import Tensor


class InverseBottleneck(nn.Module):
    expansion = 1
    exp = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        final_relu=True
    ):
        super(InverseBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("InverseBottleneck only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in InverseBottleneck")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes * self.exp, 1, 1, 0)
        self.bn1 = norm_layer(planes * self.exp)
        self.relu1 = nn.ReLU(inplace=True)        
        self.conv2 = nn.Conv2d(
                    planes * self.exp,
                    planes * self.exp,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=planes * self.exp,
                    bias=False,
                    dilation=1,
                )
        self.bn2 = norm_layer(planes * self.exp)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes * self.exp, planes, 1, 1, 0)
        self.bn3 = norm_layer(planes)
        self.relu3 = None
        if final_relu:
            self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.relu3:
            out = self.relu3(out)

        return out


class CrossM(nn.Module):
    def __init__(self, mid_channels=64, dilation=1):
        super(CrossM, self).__init__()
        self.cross_l = ConvModule(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1, 
                    padding=dilation, 
                    dilation=dilation,
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))
        self.proj_l = InverseBottleneck(
                    mid_channels,
                    mid_channels)

    def forward(self, high_x1, low_x2):
        h = F.relu(self.cross_l(high_x1) + low_x2)
        h = self.proj_l(h)
        return h


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
        self.proj_l = ConvModule(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1, 
                    padding=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))

    def forward(self, x1, x2):
        y = F.relu(x1 + self.delta_l(x2))
        z = self.proj_l(y)
        return z


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
class UDMDirLite(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64, out_channels=64, delta_mode=True, pan_mode=True, deepen_factor=1.0, widen_factor=1.0, out_strides=[16,32]):
        super(UDMDirLite, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 32
        self.block = InverseBottleneck
        self.layers = [1, 1, 1, 2, 2, 2, 4, 4]

        self.delta_mode = delta_mode
        self.pan_mode = pan_mode
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = self.norm_layer(32, eps=1e-5, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.block.exp = 3
        self.layer1 = self.make_layer(self.block, 32, self.layers[0], stride=1)
        self.layer2 = self.make_layer(self.block, 32, self.layers[1], stride=2)
        self.block.exp = 6
        self.layer3 = self.make_layer(self.block, 32, self.layers[2], stride=1)
        self.layer4 = self.make_layer(self.block, 64, self.layers[3], stride=2)
        self.block.exp = 4
        self.layer5 = self.make_layer(self.block, 96, self.layers[4], stride=1)
        self.layer6_pre = self.make_layer(self.block, 128, 1, stride=2)
        self.layer6 = self.make_layer(self.block, 128, self.layers[5]-1, stride=1)    # 
        self.block.exp = 6
        self.layer7 = self.make_layer(self.block, 128, self.layers[6], stride=1)
        self.layer8_pre = self.make_layer(self.block, 160, 1, stride=2)
        self.layer8 = self.make_layer(self.block, 160, self.layers[7]-1, stride=1)    # 

        # 1/4 ,  1/8, 1/16, 1/32
        # (dilation(k-1)+1-stride)/2
        self.down4_to_16 = \
                ConvModule(
                    mid_channels, 
                    128,
                    kernel_size=5,
                    stride=4, 
                    padding=3, 
                    dilation=2, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    mid_channels, 
                    160,
                    kernel_size=5,
                    stride=8, 
                    padding=2, 
                    dilation=2, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))

        self.compression8_to_4 = \
                ConvModule(
                    96, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    128, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))
        self.compression32_to_4 = \
                ConvModule(
                    160, 
                    mid_channels, 
                    kernel_size=1, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN'))

        self.layer4_1 = nn.Sequential(
            ConvModule(
                32,
                mid_channels,
                kernel_size=3,
                padding=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN'),
            )
        )
        self.layer4_2 = CrossM(mid_channels, 1)
        self.layer4_3 = CrossM(mid_channels, 2)
        self.layer4_4 = CrossM(mid_channels, 2)

        # appm
        self.appm = APPM(in_channels=160, mid_channels=64, out_channels=160)

        self.out_strides = out_strides

        # if self.delta_mode:
        #     self.delta_conv_1 = DeltaM(mid_channels)
        #     self.delta_conv_2 = DeltaM(mid_channels)

        if self.pan_mode:
            downsample_convs = list()
            pan_blocks = list()
            lateral_convs = list()

            # 8,16,32
            lateral_channels = [96, 128, 160]
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
        x = self.relu1(self.bn1(self.conv1(x)))  # 32 * h/2 * w/2
        x = self.layer1(x)
        x32 = self.layer2(x)    # 32 * h/4 * w/4
        x32 = self.layer3(x32)  # 64 * h/4 * w/4

        out_layer_1 = self.layer4_1(x32)

        x16 = self.layer4(x32)  # 64 * h/8 * w/8
        x16 = self.layer5(x16)  # 96 * h/8 * w/8

        compression8_to_4 = self.compression8_to_4(x16)
        out_layer_2 = \
            self.layer4_2(
                out_layer_1,
                F.interpolate(
                    compression8_to_4, scale_factor=2, mode='bilinear'
                )
            )

        x8 = self.layer6_pre(x16)       # 128 * h/16 * w/16
        x8 = F.relu(x8 + self.down4_to_16(out_layer_2))
        x8 = self.layer6(x8)            # 128 * h/16 * w/16
        x8 = self.layer7(x8)            # 128 * h/16 * w/16

        compression16_to_4 = self.compression16_to_4(x8)
        out_layer_3 = \
            self.layer4_3(
                out_layer_2,
                F.interpolate(
                    compression16_to_4, scale_factor=4, mode='bilinear'
                )
            )

        x4 = self.layer8_pre(x8)        # 160 * h/32 * w/32
        x4 = F.relu(x4 + self.down4_to_32(out_layer_3))
        x4 = self.layer8(x4)            # 160 * h/32 * w/32

        # appm
        x4 = self.appm(x4)
        compression32_to_4 = self.compression32_to_4(x4)

        out_layer_4 = \
            self.layer4_4(
                out_layer_3,
                F.interpolate(
                    compression32_to_4, scale_factor=8, mode='bilinear'
                )
            )

        out_list = []
        # if self.delta_mode:
        #     # base
        #     out_list.append(out_layer_4)

        #     # fix with delta
        #     out_layer_4 = self.delta_conv_1(out_list[-1], out_layer_3)
        #     out_list.append(out_layer_4)

        #     # fix with delta
        #     out_layer_4 = self.delta_conv_2(out_list[-1], out_layer_2)
        #     out_list.append(out_layer_4)

        if self.pan_mode:
            out_list = []
            # 1/4
            out = out_layer_4
            out_list.append(out)

            # 1/8
            downsample_feat = self.downsample_convs[0](out_list[-1])  # Conv
            out = self.pan_blocks[0](
                torch.cat([downsample_feat, self.lateral_convs[0](x16)], axis=1))
            out_list.append(out)

            # 1/16
            downsample_feat = self.downsample_convs[1](out_list[-1])
            out = self.pan_blocks[1](
                torch.cat([downsample_feat, self.lateral_convs[1](x8)], axis=1)
            )
            out_list.append(out)

            # 1/32
            downsample_feat = self.downsample_convs[2](out_list[-1])
            out = self.pan_blocks[2](
                torch.cat([downsample_feat, self.lateral_convs[2](x4)], axis=1)
            )
            out_list.append(out)

            # filter out list
            index_map = {4: 0, 8: 1, 16: 2, 32: 3}
            filter_out_list = [out_list[index_map[s]] for s in self.out_strides]
            return filter_out_list

        return out_list

    def make_layer(self, block, planes, blocks, stride=1, no_final_relu=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1 and no_final_relu:
                layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, final_relu=False))
            else:
                layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)
