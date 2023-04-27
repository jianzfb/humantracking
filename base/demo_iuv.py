import sys
sys.path.append('/root/workspace/antgo')
from utils.renderer import PyRenderer, IUV_Renderer
from utils.iuvmap import *
from utils.geometry import *
from base_dataset import BaseDataset
from models.smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14

batch_size = 2
train_ds = BaseDataset('coco-full', batch_size=batch_size, is_train=True)
smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=batch_size,
            create_transl=False
        )


img_res = 224
focal_length = 5000.
dp_heatmap_size = 56
iuv_maker = IUV_Renderer(output_size=dp_heatmap_size)

for input_batch in train_ds:
    gt_pose = input_batch['pose']       # SMPL pose parameters
    gt_betas = input_batch['betas']     # SMPL beta parameters
    gt_joints = input_batch['pose_3d']  # 3D pose
    gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints

    gt_out = smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
    opt_vertices = gt_out.vertices
    opt_joints = gt_out.joints  # smpl 获得的关键点

    gt_keypoints_2d_orig = gt_keypoints_2d.clone()
    gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

    opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=focal_length, img_size=img_res)
    gt_cam_t_nr = opt_cam_t.detach().clone()
    gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
    gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
    gt_camera[:, 0] = (2. * focal_length / img_res) / gt_cam_t_nr[:, 2]
    iuv_image_gt = torch.zeros((batch_size, 3, dp_heatmap_size, dp_heatmap_size))

    iuv_image = iuv_maker.verts2iuvimg(gt_out.vertices, cam=camera)  # [B, 3, 56, 56]
    uvia_list = iuv_img2map(iuv_image)