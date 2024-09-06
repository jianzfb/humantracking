import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn import ConvModule,DepthwiseSeparableConvModule
from antgo.framework.helper.models.segmentation.head.aspp_head import *


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
    def __init__(self):
        super(CrossM, self).__init__()
        self.conv_L0 = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))
        self.conv1_L1 = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))
        self.conv2_L1 = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))

        self.conv1_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=2,
                    dilation=2,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))
        self.conv2_L2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=2, 
                    dilation=2,
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))
        self.conv_L3 = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))

    def forward(self, high_x1, low_x2):
        h = F.relu(self.conv_L0(high_x1) + low_x2)

        h1 = self.conv1_L1(h) # branch1
        h1 = self.conv2_L1(h1)

        h2 = self.conv1_L2(h) # branch2
        h2 = self.conv2_L2(h2)
        h = F.relu(h1 + h2)

        h = self.conv_L3(h)
        return h


class DeltaM(nn.Module):
    def __init__(self):
        super(DeltaM, self).__init__()
        self.delta_l = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))

        self.conv_l1 = ConvModule(
                    64,
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))

        self.conv_l2 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=2,
                    dilation=2,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))

        self.conv_l3 = ConvModule(
                    64, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=4,
                    dilation=4,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=dict(type='BN'))

        self.conv_proj = ConvModule(
                    192, 
                    64,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=None,
                    norm_cfg=dict(type='BN')) 

    def forward(self, x1, x2):
        y = F.relu(x1 + self.delta_l(x2))

        z_1 = self.conv_l1(y)
        z_2 = self.conv_l2(y + z_1)
        z_3 = self.conv_l3(y + z_2)

        z = self.conv_proj(torch.cat([z_1, z_2, z_3], dim=1))
        return z


@BACKBONES.register_module()
class BackboneDeltaDDRM(nn.Module):
    def __init__(self, in_channels=1):
        super(BackboneDeltaDDRM, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 32
        self.block = InverseBottleneck
        self.layers = [1, 1, 1, 2, 2, 2, 4, 4]

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
                    64, 
                    128,
                    kernel_size=5,
                    stride=4, 
                    padding=(int)((1*(5-1)+1-4)//2+1), 
                    dilation=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    64, 
                    160,
                    kernel_size=5,
                    stride=8, 
                    padding=(int)(1), 
                    dilation=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))             

        self.compression8_to_4 = \
                ConvModule(
                    96, 
                    64, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    128, 
                    64, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.compression32_to_4 = \
                ConvModule(
                    160, 
                    64, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))

        self.layer4_1 = nn.Sequential(
            ConvModule(
                32,
                64,
                kernel_size=3,
                padding=1,
                act_cfg=None,
                norm_cfg=dict(type='BN'),
            )
        )

        self.layer4_2 = CrossM()
        self.layer4_3 = CrossM()
        self.layer4_4 = CrossM()

        self.layer4_delta_1 = DeltaM()
        self.layer4_delta_2 = DeltaM()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 32 * h/2 * w/2
        x = self.layer1(x)
        x32 = self.layer2(x)    # 32 * h/4 * w/4
        x32 = self.layer3(x32)  # 64 * h/4 * w/4

        out_layer_1 = self.layer4_1(x32)

        x16 = self.layer4(x32)  # 64 * h/8 * w/8
        x16 = self.layer5(x16)  # 96 * h/8 * w/8

        out_layer_2 = \
            self.layer4_2(
                out_layer_1,
                F.interpolate(
                    self.compression8_to_4(x16), scale_factor=2, mode='bilinear'
                )
            )

        x8 = self.layer6_pre(x16)       # 128 * h/16 * w/16
        x8 = F.relu(x8 + self.down4_to_16(out_layer_2))
        x8 = self.layer6(x8)            # 128 * h/16 * w/16
        x8 = self.layer7(x8)            # 128 * h/16 * w/16

        out_layer_3 = \
            self.layer4_3(
                out_layer_2,
                F.interpolate(
                    self.compression16_to_4(x8), scale_factor=4, mode='bilinear'
                )
            )

        x4 = self.layer8_pre(x8)        # 160 * h/32 * w/32
        x4 = F.relu(x4 + self.down4_to_32(out_layer_3))
        x4 = self.layer8(x4)            # 160 * h/32 * w/32

        out_layer_4 = \
            self.layer4_4(
                out_layer_3,
                F.interpolate(
                    self.compression32_to_4(x4), scale_factor=8, mode='bilinear'
                )
            )

        # base
        out_list = [out_layer_4]

        # refine 1
        out_layer_delta_1 = self.layer4_delta_1(out_layer_4, out_layer_3)
        out_layer_4 = out_layer_4 + out_layer_delta_1
        out_list.append(out_layer_4)

        # refine 2
        out_layer_delta_2 = self.layer4_delta_2(out_layer_4, out_layer_2)
        out_layer_4 = out_layer_4 + out_layer_delta_2
        out_list.append(out_layer_4)
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
