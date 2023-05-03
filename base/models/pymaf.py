import torch
import torch.nn as nn
import numpy as np
from .pose_resnet import get_resnet_encoder
from antgo.framework.helper.models.builder import MODELS, build_model
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .iuv_predictor import IUV_predict_layer
from antgo.framework.helper.runner import BaseModule
from core.path_config import *
from core.constants import *
from utils.renderer import PyRenderer, IUV_Renderer
from utils.iuvmap import iuv_img2map, iuv_map2img
import torch.nn.functional as F


import logging
logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1

class Regressor(nn.Module):
    def __init__(self, batch_size, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=batch_size,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output


@MODELS.register_module()
class PyMAF(BaseModule):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, train_cfg=None, **kwargs):
        super().__init__()
        self.global_mode = not train_cfg.MAF_ON
        self.feature_extractor = get_resnet_encoder(train_cfg.POSE_RES_MODEL, global_mode=self.global_mode)

        self.train_cfg = train_cfg
        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        self.deconv_with_bias = train_cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            train_cfg.RES_MODEL.NUM_DECONV_LAYERS,
            train_cfg.RES_MODEL.NUM_DECONV_FILTERS,
            train_cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(train_cfg.N_ITER):
            self.maf_extractor.append(MAF_Extractor(train_cfg.MLP_DIM))
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * train_cfg.MLP_DIM[-1]
        
        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * train_cfg.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(train_cfg.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            self.regressor.append(Regressor(batch_size=train_cfg.batch_size, feat_dim=ref_infeat_dim, smpl_mean_params=SMPL_MEAN_PARAMS))

        # dp_feat_dim = 256
        # self.with_uv = train_cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        # self.criterion_keypoints = nn.MSELoss(reduction='none')
        # self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)
        # self.img_res = 224
        # self.focal_length = FOCAL_LENGTH
        # # self.iuv_maker = IUV_Renderer(output_size=self.train_cfg.DP_HEATMAP_SIZE, device=torch.device('cpu'))
        # self.smpl = self.regressor[0].smpl

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d]
        conf = conf[has_pose_3d]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl]
        gt_vertices_with_shape = gt_vertices[has_smpl]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl]
        pred_betas_valid = pred_betas[has_smpl]
        gt_betas_valid = gt_betas[has_smpl]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def body_uv_losses(self, u_pred, v_pred, index_pred, ann_pred, uvia_list, has_iuv=None):
        batch_size = index_pred.size(0)
        device = index_pred.device

        Umap, Vmap, Imap, Annmap = uvia_list

        if has_iuv is not None:
            if torch.sum(has_iuv.float()) > 0:
                u_pred = u_pred[has_iuv] if u_pred is not None else u_pred
                v_pred = v_pred[has_iuv] if v_pred is not None else v_pred
                index_pred = index_pred[has_iuv] if index_pred is not None else index_pred
                ann_pred = ann_pred[has_iuv] if ann_pred is not None else ann_pred
                Umap, Vmap, Imap = Umap[has_iuv], Vmap[has_iuv], Imap[has_iuv]
                Annmap = Annmap[has_iuv] if Annmap is not None else Annmap
            else:
                return (torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device))

        Itarget = torch.argmax(Imap, dim=1)
        Itarget = Itarget.view(-1).to(torch.int64)

        index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
        index_pred = index_pred.view(-1, Imap.size(1))

        loss_IndexUV = F.cross_entropy(index_pred, Itarget)

        if self.train_cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0:
            loss_U = F.smooth_l1_loss(u_pred[Imap > 0], Umap[Imap > 0], reduction='sum') / batch_size
            loss_V = F.smooth_l1_loss(v_pred[Imap > 0], Vmap[Imap > 0], reduction='sum') / batch_size

            loss_U *= self.train_cfg.LOSS.POINT_REGRESSION_WEIGHTS
            loss_V *= self.train_cfg.LOSS.POINT_REGRESSION_WEIGHTS
        else:
            loss_U, loss_V = torch.zeros(1).to(device), torch.zeros(1).to(device)

        if ann_pred is None:
            loss_segAnn = None
        else:
            Anntarget = torch.argmax(Annmap, dim=1)
            Anntarget = Anntarget.view(-1).to(torch.int64)
            ann_pred = ann_pred.permute([0, 2, 3, 1]).contiguous()
            ann_pred = ann_pred.view(-1, Annmap.size(1))
            loss_segAnn = F.cross_entropy(ann_pred, Anntarget)

        return loss_U, loss_V, loss_IndexUV, loss_segAnn
    
    def loss(self, batch_size, preds_dict, gt_dict):
        gt_keypoints_2d = gt_dict['keypoints'] # 2D keypoints
        gt_pose = gt_dict['pose']       # SMPL pose parameters
        gt_betas = gt_dict['betas']     # SMPL beta parameters
        gt_joints = gt_dict['pose_3d']  # 3D pose
        has_smpl = gt_dict['has_smpl'].to(torch.bool) # flag that indicates whether SMPL parameters are valid
        has_pose_3d = gt_dict['has_pose_3d'].to(torch.bool) # flag that indicates whether 3D pose is valid        
        
        # 准备数据
        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        opt_vertices = gt_out.vertices
        opt_joints = gt_out.joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.img_res)
        valid_fit = has_smpl
        gt_cam_t_nr = opt_cam_t.detach().clone()
        gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
        gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
        gt_camera[:, 0] = (2. * self.focal_length / self.img_res) / gt_cam_t_nr[:, 2]
        iuv_image_gt = torch.zeros((batch_size, 3, self.train_cfg.DP_HEATMAP_SIZE, self.train_cfg.DP_HEATMAP_SIZE)).to(gt_keypoints_2d.device)
        if torch.sum(valid_fit.float()) > 0:
            iuv_image_gt[valid_fit] = self.iuv_maker.verts2iuvimg(opt_vertices[valid_fit], cam=gt_camera[valid_fit])  # [B, 3, 56, 56]
        uvia_list = iuv_img2map(iuv_image_gt)


        # 计算损失
        loss_dict = {}
        dp_out = preds_dict['dp_out']
        for i in range(len(dp_out)):
            r_i = i - len(dp_out)

            u_pred, v_pred, index_pred, ann_pred = dp_out[r_i]['predict_u'], dp_out[r_i]['predict_v'], dp_out[r_i]['predict_uv_index'], dp_out[r_i]['predict_ann_index']
            if index_pred.shape[-1] == iuv_image_gt.shape[-1]:
                uvia_list_i = uvia_list
            else:
                iuv_image_gt_i = F.interpolate(iuv_image_gt, u_pred.shape[-1], mode='nearest')
                uvia_list_i = iuv_img2map(iuv_image_gt_i)

            loss_U, loss_V, loss_IndexUV, loss_segAnn = self.body_uv_losses(u_pred, v_pred, index_pred, ann_pred,
                                                                            uvia_list_i, valid_fit)
            loss_dict[f'loss_U{r_i}'] = loss_U
            loss_dict[f'loss_V{r_i}'] = loss_V
            loss_dict[f'loss_IndexUV{r_i}'] = loss_IndexUV
            loss_dict[f'loss_segAnn{r_i}'] = loss_segAnn

        len_loop = len(preds_dict['smpl_out'])
        for l_i in range(len_loop):
            if l_i == 0:
                # initial parameters (mean poses)
                continue
            pred_rotmat = preds_dict['smpl_out'][l_i]['rotmat']
            pred_betas = preds_dict['smpl_out'][l_i]['theta'][:, 3:13]
            pred_camera =  preds_dict['smpl_out'][l_i]['theta'][:, :3]

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                                    global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                    pred_camera[:,2],
                                    2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)

            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                    rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                    translation=pred_cam_t,
                                                    focal_length=self.focal_length,
                                                    camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

            # Compute loss on SMPL parameters
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, valid_fit)
            loss_regr_pose *= self.train_cfg.LOSS.POSE_W
            loss_regr_betas *= self.train_cfg.LOSS.SHAPE_W
            loss_dict['loss_regr_pose_{}'.format(l_i)] = loss_regr_pose
            loss_dict['loss_regr_betas_{}'.format(l_i)] = loss_regr_betas

            # Compute 2D reprojection loss for the keypoints
            if self.train_cfg.LOSS.KP_2D_W > 0:
                loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                    self.train_cfg.openpose_train_weight,
                                                    self.train_cfg.gt_train_weight) * self.train_cfg.LOSS.KP_2D_W
                loss_dict['loss_keypoints_{}'.format(l_i)] = loss_keypoints

            # Compute 3D keypoint loss
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d) * self.train_cfg.LOSS.KP_3D_W
            loss_dict['loss_keypoints_3d_{}'.format(l_i)] = loss_keypoints_3d

            # Per-vertex loss for the shape
            if self.train_cfg.LOSS.VERT_W > 0:
                loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit) * self.train_cfg.LOSS.VERT_W
                loss_dict['loss_shape_{}'.format(l_i)] = loss_shape

            # Camera
            # force the network to predict positive depth values
            loss_cam = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
            loss_dict['loss_cam_{}'.format(l_i)] = loss_cam

        for key in loss_dict:
            if len(loss_dict[key].shape) > 0:
                loss_dict[key] = loss_dict[key][0]

        return loss_dict        
    
    def forward(self, image, J_regressor=None, **kwargs):
        batch_size = image.shape[0]

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(image)

        assert self.train_cfg.N_ITER >= 0 and self.train_cfg.N_ITER <= 3
        if self.train_cfg.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif self.train_cfg.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif self.train_cfg.N_ITER == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        # 计算特征
        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat, J_regressor=J_regressor)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # parameter predictions
        for rf_i in range(self.train_cfg.N_ITER):
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())

            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)
            else:
                pred_smpl_verts = smpl_output['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0), pred_smpl_verts) # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds) # [B, 431 * n_feat]

            smpl_output = self.regressor[rf_i](ref_feature, pred_pose, pred_shape, pred_cam, n_iter=1, J_regressor=J_regressor)
            out_list['smpl_out'].append(smpl_output)

        # iuv_out_dict = self.dp_head(s_feat)
        # out_list['dp_out'].append(iuv_out_dict)

        # # 计算损失
        # loss_dict = self.loss(batch_size, out_list, kwargs)
        # return loss_dict
        return out_list
