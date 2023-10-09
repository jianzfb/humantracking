import torch
import torchvision
from torch import nn
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Pose_ViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Pose_ViT, self).__init__()
        # Load pretrained model
        self.ViT = torchvision.models.vision_transformer.vit_b_16(pretrained=pretrained)

        # Set the same parameters for the corresponding model
        self.patch_size = 16
        self.hidden_dim = 768
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.conv_proj = nn.Sequential(self.ViT.conv_proj)
        self.encoder = nn.Sequential(self.ViT.encoder)

        # These two outputs correspond to mesh aligned inputs 
        self.patch_token_out = nn.Linear(self.hidden_dim, num_classes)
        self.downsample = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1)
    
    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        x = self.patch_token_out(x)
    
        patch_token = x.clone()
        # # Extract feature part
        # s_feat = x[:, 1:]
        # s_feat_list = repeat(s_feat, 'b n p -> b p n')
        # b_s, p_s, n_s = s_feat_list.shape 
        # s_feat_list = s_feat_list.reshape(b_s, p_s, int(math.sqrt(n_s)), int(math.sqrt(n_s)))
        # s_feat_list = self.downsample(s_feat_list)

        # Extract cls part
        g_feat = x[:, 0]

        return patch_token, g_feat


def get_vit_encoder():
    model = Pose_ViT(num_classes=2048, pretrained=True)
    return model
