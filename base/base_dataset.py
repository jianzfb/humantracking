from antgo.framework.helper.dataset.builder import DATASETS
from core import path_config
from core import constants 
import numpy as np
import torch
import cv2
import os
from torchvision.transforms import Normalize
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, transform_pts, rot_aa
from models.smpl import SMPL
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@DATASETS.register_module()
class BaseDataset(object):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/path_config.py.
    """

    def __init__(self, 
                 dataset, 
                 batch_size,
                 ignore_3d=False, 
                 use_augmentation=True, 
                 is_train=True, 
                 noise_factor=0.4,
                 rot_factor=30,
                 scale_factor=0.25
                 ):
        self.dataset = dataset
        self.is_train = is_train
        self.noise_factor = noise_factor
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.is_debug = False
        
        self.img_dir = path_config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        if not is_train and dataset == 'h36m-p2':
            self.data = np.load(path_config.DATASET_FILES[is_train]['h36m-p2-mosh'], allow_pickle=True)
        else:
            self.data = np.load(path_config.DATASET_FILES[is_train][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']
        self.dataset_dict = {dataset: 0}
        logger.info('len of {}: {}'.format(self.dataset, len(self.imgname)))

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)      # (N, 72)
            self.betas = self.data['shape'].astype(np.float)    # (N, 10)

            ################# generate final_fits file in case of it is missing #################
            # import os
            # params_ = np.concatenate((self.pose, self.betas), axis=-1)
            # out_file_fit = os.path.join('data/final_fits', self.dataset)
            # if not os.path.exists('data/final_fits'):
            #     os.makedirs('data/final_fits')
            # np.save(out_file_fit, params_)
            # raise ValueError('Please copy {}.npy file to data/final_fits, and delete this code block.'.format(self.dataset))
            ########################################################################

            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        # Get SMPL 2D keypoints
        try:
            self.smpl_2dkps = self.data['smpl_2dkps']
            self.has_smpl_2dkps = 1
        except KeyError:
            self.has_smpl_2dkps = 0

        # Get gt 3D jonits, if available
        try:
            self.joints_3d = self.data['S']
            self.has_joints_3d = 1
        except KeyError:
            self.has_joints_3d = 0
        if ignore_3d:
            self.has_joints_3d = 0

        # Get gt 2D joints
        try:
            joints_2d_gt = self.data['part']
        except KeyError:
            joints_2d_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            joints_2d_openpose = self.data['openpose']
        except KeyError:
            joints_2d_openpose = np.zeros((len(self.imgname), 25, 3))

        # openpose + smpl 
        self.joints_2d = np.concatenate([joints_2d_openpose, joints_2d_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
    
        self.length = self.scale.shape[0]

        self.smpl = SMPL(path_config.SMPL_MODEL_DIR,
                    batch_size=batch_size,
                    create_transl=False)

        self.faces = self.smpl.faces
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling

        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2,0,1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_smpl=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f, is_smpl=False):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S, is_smpl)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def debug_pose_show(self, image, joints, pose, beta):
        pass

    def debug_joints2d_show(self, image, joints2d):
        h,w = image.shape[:2]
        for x,y,visible in joints2d:
            if int(visible):
                cv2.circle(image, (int((x+1)/2.0 * w), int((y+1)/2.0 * h)), (2), (255,0,0), 5)

        if not os.path.exists('./debug'):
            os.mkdir('./debug')
        cv2.imwrite(f'./debug/show.png', image)

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = os.path.join(self.img_dir, self.imgname[index])
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]

        kp_is_smpl = True if self.dataset == 'surreal' else False

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            pose = self.pose_processing(pose, rot, flip)
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        if self.is_debug:
            cv2.circle(img, (int(center[0]), int(center[1])), (2), (0,255,0), 2)
            cv2.imwrite('./debug/debug_center.png', img.astype(np.uint8))

            caice_half_width = 200*scale/2
            caice_half_height = 200*scale/2
            cv2.rectangle(img, (int(center[0]-caice_half_width), int(center[1]-caice_half_height)), (int(center[0]+caice_half_width), int(center[1]+caice_half_height)), (0,0,255), 3)
            cv2.imwrite('./debug/debug_bbox.png', img.astype(np.uint8))

        # Store image before normalization to use it in visualization
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        
        item['image'] = self.normalize_img(torch.from_numpy(img).float())
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()

        # TODO, 暂时不考虑顶点
        # if self.has_smpl[index]:
        #     betas_th = item['betas'].unsqueeze(0)
        #     pose_th = item['pose'].unsqueeze(0)
        #     smpl_out = self.smpl(betas=betas_th, body_pose=pose_th[:, 3:], global_orient=pose_th[:, :3], pose2rot=True)
        #     verts = smpl_out.vertices[0]
        #     item['verts'] = verts
        # else:
        #     item['verts'] = torch.zeros(6890, 3, dtype=torch.float32)

        # Get 2D SMPL joints
        # 2D keypionts (from SMPL)
        if self.has_smpl_2dkps:
            smpl_2dkps = self.smpl_2dkps[index].copy()
            smpl_2dkps = self.j2d_processing(smpl_2dkps, center, sc * scale, rot, f=0)
            smpl_2dkps[smpl_2dkps[:, 2] == 0] = 0
            if flip:
                smpl_2dkps = smpl_2dkps[constants.SMPL_JOINTS_FLIP_PERM]
                smpl_2dkps[:, 0] = - smpl_2dkps[:, 0]
            item['smpl_2dkps'] = torch.from_numpy(smpl_2dkps).float()
        else:
            item['smpl_2dkps'] = torch.zeros(24, 3, dtype=torch.float32)

        # Get 3D pose, if available
        # 3D keypoints (from SMPL)
        if self.has_joints_3d:
            S = self.joints_3d[index].copy()
            item['joints_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip, kp_is_smpl)).float()
        else:
            item['joints_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        # openpose(25) + smpl(24)
        joints_2d = self.joints_2d[index].copy()
        item['joints_2d'] = torch.from_numpy(self.j2d_processing(joints_2d, center, sc*scale, rot, flip, kp_is_smpl)).float()

        # debug joints_2d
        if self.is_debug:
            # 验证2D关键点正确性
            self.debug_joints2d_show(np.transpose(img*255,[1,2,0]).astype(np.uint8), item['joints_2d'])

        item['has_smpl'] = self.has_smpl[index]
        item['has_joints_3d'] = self.has_joints_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index

        # item['dataset_name'] = self.dataset
        # try:
        #     item['maskname'] = self.maskname[index]
        # except AttributeError:
        #     item['maskname'] = ''
        # try:
        #     item['partname'] = self.partname[index]
        # except AttributeError:
        #     item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)

if __name__ == '__main__':
    bd = BaseDataset('coco-full', 2, is_train=True)
    for i in range(len(bd)):
        data = bd[i]
        print(i)