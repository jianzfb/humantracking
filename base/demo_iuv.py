import sys
sys.path.append('/Users/bytedance/Downloads/workspace/antgo')
from utils.renderer import PyRenderer, IUV_Renderer
from utils.iuvmap import *
from utils.geometry import *
from base_dataset import BaseDataset
from models.smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from core.constants import *
import cv2

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
iuv_maker = IUV_Renderer(output_size=dp_heatmap_size, device=torch.device('cpu'))


def _debug_show(image, iuv_image):
    image = image * torch.from_numpy(np.array(IMG_NORM_STD)).view(3,1,1) + torch.from_numpy(np.array(IMG_NORM_MEAN)).view(3,1,1)
    image = image * 255
    image = torch.permute(image, (1,2,0))

    iuv_image = torch.permute(iuv_image.view(3,56,56)*255, (1,2,0))
    cv2.imwrite('./a.png', image.detach().cpu().numpy().astype(np.uint8))
    cv2.imwrite('./b.png', iuv_image.detach().cpu().numpy().astype(np.uint8))

def _debug_joints(image, joints):
    joints = joints[0]
    joint_num = joints.shape[0]
    image = image * torch.from_numpy(np.array(IMG_NORM_STD)).view(3,1,1) + torch.from_numpy(np.array(IMG_NORM_MEAN)).view(3,1,1)
    image = image * 255
    image = torch.permute(image, (1,2,0))
    
    image = image.detach().cpu().numpy().astype(np.uint8)
    
    for joint_i in range(joint_num):
        x,y = joints[joint_i].detach().cpu().numpy()
        image = cv2.circle(image, (int(x),int(y)), 2, (0,0,255), 2)
    
    cv2.imwrite('./c.png', image)


for input_batch in train_ds:
    # Nx72
    gt_pose = torch.unsqueeze(input_batch['pose'], 0)               # SMPL pose parameters
    # Nx10
    gt_betas = torch.unsqueeze(input_batch['betas'], 0)             # SMPL beta parameters
    # Nx24x4
    gt_joints = torch.unsqueeze(input_batch['pose_3d'], 0)          # 3D pose
    # Nx49x3
    gt_keypoints_2d = torch.unsqueeze(input_batch['keypoints'], 0)  # 2D keypoints
    
    gt_out = smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
    opt_vertices = gt_out.vertices
    opt_joints = gt_out.joints  # smpl 获得的关键点

    # De-normalize 2D keypoints from [-1,1] to pixel space
    gt_keypoints_2d_orig = gt_keypoints_2d.clone()
    gt_keypoints_2d_orig[:, :, :-1] = 0.5 * img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

    # 基于假设的焦距、成像大小、投影2D点估算平移向量
    opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=focal_length, img_size=img_res)
    
    # step1: 测试IUV信息
    gt_cam_t_nr = opt_cam_t.detach().clone()
    gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
    gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
    gt_camera[:, 0] = (2. * focal_length / img_res) / gt_cam_t_nr[:, 2]
    iuv_image = iuv_maker.verts2iuvimg(gt_out.vertices, cam=gt_camera)  # [B, 3, 56, 56]
    
    # debug image show
    _debug_show(input_batch['img'], iuv_image)
    
    # 
    uvia_list = iuv_img2map(iuv_image)   
    
    # step2: 测试SMPL投影
    camera_center = torch.zeros(1, 2, device='cpu')
    camera_center[0,0] = img_res/2.
    camera_center[0,1] = img_res/2.
    t = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
    t[:,0] = gt_cam_t_nr[:, 0]
    t[:,1] = gt_cam_t_nr[:, 1]
    t[:,2] = gt_cam_t_nr[:, 2]
    pred_keypoints_2d = perspective_projection(
        opt_joints,
        rotation=torch.eye(3, device='cpu').unsqueeze(0).expand(1, -1, -1),
        translation=t,
        focal_length=focal_length,
        camera_center=camera_center)

    # 将投影点在图像上可视化
    _debug_joints(input_batch['img'], pred_keypoints_2d)
    print('s')