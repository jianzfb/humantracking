import torch
import torch.nn as nn
import numpy as np
from antgo.framework.helper.models.builder import MODELS, build_model
from utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .vit_model import get_vit_encoder
from .cross_attention_fusion import cross_attention, Transformer_Block
from antgo.framework.helper.runner import BaseModule
import cv2
import logging
logger = logging.getLogger(__name__)


class PoseCorrectModule(nn.Module):
    def __init__(self, embed_dims, smpl_mean_params):
        super().__init__()

        # The part of the fusion of pred_joints and patch_token
        self.patch_mlp = nn.Linear(2048, 3)
        self.cross_attention = cross_attention
        self.self_attention = Transformer_Block(dim=2048)

        # info guided cross-attention

        # error guided cross-attention
        # self.condition_attention = ConditionalAttention(input_dim=2048)

        # SMPL参数编码
        self.pose_embed = nn.Linear(6, embed_dims)      # rot6d
        self.shape_embed = nn.Linear(10, embed_dims)    # 
        self.cam_embed = nn.Linear(3, embed_dims)       # 

        # SMPL MPL
        self.pose_mpl = nn.Linear(embed_dims, 6)
        self.shape_mpl = nn.Linear(embed_dims, 10)
        self.cam_mpl = nn.Linear(embed_dims, 3)

        # 关键点编码网络
        self.joint_embed = nn.Linear(3, embed_dims)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, smpl_token, patch_token, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = smpl_token.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
            init_pose = init_pose.reshape(batch_size, -1, 6)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1, -1)

        # step1: 将init_pose, init_shape, init_cam编码, 并修正smpl_token
        init_pose_encoder = self.pose_embed(init_pose)
        init_shape_encoder = self.shape_embed(init_shape)
        init_cam_encoder = self.cam_embed(init_cam)
        init_token = torch.cat([init_pose_encoder, init_shape_encoder, init_cam_encoder], 1)

        smpl_token = self.cross_attention(smpl_token, init_token)
        smpl_token = self.self_attention(smpl_token)

        # step2: kinetic tree info 
        # kinetic tree attention

        # step3: 基于误差修正smpl_token
        for i in range(n_iter):
            pose_token = smpl_token[:, :24]
            shape_token= smpl_token[:, 24:25]
            cam_token = smpl_token[:, 25:]

            # 将token转成SMPL参数
            pred_pose_param = self.pose_mpl(pose_token).reshape(batch_size, -1)
            pred_shape_param = self.shape_mpl(shape_token).reshape(batch_size, -1)
            pred_cam_param = self.cam_mpl(cam_token).reshape(batch_size, -1)

            # 获取旋转矩阵，并输入SMPL
            pred_rotmat = rot6d_to_rotmat(pred_pose_param).view(batch_size, 24, 3, 3)
            pred_output = self.smpl(
                betas=pred_shape_param,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False
            )

            pred_vertices = pred_output.vertices
            pred_joints_3d = pred_output.joints
            pred_smpl_joints = pred_output.smpl_joints
            pred_joints_2d = projection(pred_joints_3d, pred_cam_param)
            # pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
            # if J_regressor is not None:
            #     pred_joints = torch.matmul(J_regressor, pred_vertices)
            #     pred_pelvis = pred_joints[:, [0], :].clone()
            #     pred_joints = pred_joints[:, H36M_TO_J14, :]
            #     pred_joints = pred_joints - pred_pelvis

            # 对pred_joints编码为token
            pred_joints_encoder = self.joint_embed(pred_joints_3d)

            # # condition attention
            # smpl_token_length = smpl_token.shape[1]
            # smpl_token = self.condition_attention(smpl_token, pred_joints_encoder[:, :smpl_token_length, :], patch_token[:, :smpl_token_length, :])

            # self attention
            smpl_token = self.self_attention(smpl_token)

        # 输出
        pose_token = smpl_token[:, :24]
        shape_token= smpl_token[:, 24:25]
        cam_token = smpl_token[:, 25:]

        pred_pose_param = self.pose_mpl(pose_token).reshape(batch_size, -1)
        pred_shape_param = self.shape_mpl(shape_token).reshape(batch_size, -1)
        pred_cam_param = self.cam_mpl(cam_token).reshape(batch_size, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose_param).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
                betas=pred_shape_param,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False
            )

        pred_vertices = pred_output.vertices
        pred_joints_3d = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_joints_2d = projection(pred_joints_3d, pred_cam_param)

        output = {
            'verts'  : pred_vertices,
            'kp_2d'  : pred_joints_2d,
            'kp_3d'  : pred_joints_3d,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,             # 3x3
            'pred_cam': pred_cam_param,         # 3
            'pred_shape': pred_shape_param,     # 10
            'pred_pose': pred_pose_param,       # 24*3
        }

        return output



@MODELS.register_module()
class TransPose(BaseModule):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True, embed_dims=2048, train_cfg=None, **kwargs):
        super().__init__()
        self.feature_extractor = get_vit_encoder()
        self.cross_attention = cross_attention
        
        # 50(smpl input) = 24(pose) + 1(shape) + 1(camera)
        self.smpl_token = nn.Embedding(26, embed_dims)
        self.inplanes = 2048
        self.pose_correct = PoseCorrectModule(embed_dims=2048, smpl_mean_params=smpl_mean_params)

    def forward(self, image, **kwargs):
        x = image
        batch_size = x.shape[0]

        # spatial features and global features
        patch_token, _ = self.feature_extractor(x)
        smpl_token = self.smpl_token.weight.repeat(batch_size, 1, 1)

        # cross attention of smpl_token and patch_token
        smpl_token = self.cross_attention(smpl_token, patch_token)

        # out_list = {}
        # out_list['smpl_out'] = []
        # out_list['dp_out'] = []


        # smpl_output = self.regressor(smpl_token, patch_token, n_iter=3, J_regressor=J_regressor)
        # out_list['smpl_out'].append(smpl_output)
        # return out_list, vis_feat_list

        output = self.pose_correct(smpl_token, patch_token)

        # 计算损失
        loss_dict = {
            'loss': 0.0
        }
        return loss_dict

def pymaf_net(smpl_mean_params, pretrained=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params, pretrained)
    return model

