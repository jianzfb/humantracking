import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
import torchvision


@BACKBONES.register_module()
class ResnetTorchV(nn.Module):
    """KetNetF"""

    def __init__(self):
        super(ResnetTorchV, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x4 = self.model.layer1(x)
        x8 = self.model.layer2(x4)
        x16 = self.model.layer3(x8)
        x32 = self.model.layer4(x16)
        return [x8,x16,x32]