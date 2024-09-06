from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper import reader
from antgo.dataflow.dataset.dataset import Dataset
import numpy as np
import torch
import json
import os
import cv2


# @reader.register
# class YourDataset(Dataset):
#     def __init__(self, train_or_test="train", dir=None, ext_params=None):
#         super().__init__(train_or_test, dir, ext_params)
#         pass
    
#     @property
#     def size(self):
#         # 返回数据集大小
#         return 0
    
#     def sample(self, id):
#         # 根据id，返回对应样本
#         return {}


@reader.register
class PersonPoseDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        self.anno_list = ext_params['anno_list']
        self.joints_num = ext_params['joints_num']
        self.anno_info = []
        for anno_file in self.anno_list:
            base_folder = os.path.basename(anno_file)
            with open(anno_file, 'r') as fp:
                content = json.load(fp)
                self.anno_info.extend(content)

        self.anno_big_file = os.path.join(self.dir, 'train_with_facebox_facekeypoint_zhuohua_handbox_update4mediapipeOrder_230818.json')
        with open(self.anno_big_file, 'r') as fp:
            self.anno_big_info = json.load(fp)
        
        print(f'baidu pose dataset num {len(self.anno_big_info)}')

    @property
    def size(self):
        # 返回数据集大小
        return len(self.anno_info) + len(self.anno_big_info)
    
    def sample(self, id):
        # 根据id，返回对应样本
        if id < len(self.anno_info):
            anno_info = self.anno_info[id]
            image = cv2.imread(os.path.join(self.dir, anno_info['image']))
            mask = cv2.imread(os.path.join(self.dir, anno_info['mask']), cv2.IMREAD_GRAYSCALE)
            mask = mask/255
            mask = mask.astype(np.uint8)

            pose = np.zeros((1, self.joints_num,3), dtype=np.float32)
            bboxes = np.array([[0,0,mask.shape[1],mask.shape[0]]])
            has_pose = False
            return {
                'image': image,
                'segments': mask,
                'joints2d': pose[:,:,:2].copy(),
                'joints_vis': pose[:, :,2].astype(np.int32),
                'has_joints': has_pose,
                'has_segments': True,
                'bboxes': bboxes
            }
        else:
            id = id - len(self.anno_info)
            anno_info = self.anno_big_info[id]

            image_id = anno_info['image_id']
            image_dataset = anno_info['dataset']

            image_path = os.path.join(self.dir, 'images', image_dataset, image_id)
            image = cv2.imread(image_path)
            image_h, image_w = image.shape[:2]
            person_num = len(anno_info['keypoint_annotations'])
            random_person_i = np.random.choice(person_num)
            keys = list(anno_info['keypoint_annotations'].keys())
            random_person_key = keys[random_person_i]
            random_person_points = anno_info['keypoint_annotations'][random_person_key]
            random_person_points = np.array(random_person_points)
            x0 = np.min(random_person_points[:,0])
            y0 = np.min(random_person_points[:,1])
            x1 = np.max(random_person_points[:,0])
            y1 = np.max(random_person_points[:,1])

            x0 = int(max(0, x0-20))
            y0 = int(max(0, y0-20))
            x1 = int(min(x1+20, image_w))
            y1 = int(min(y1+20, image_h))

            random_person_points[:,0] = random_person_points[:,0] - x0
            random_person_points[:,1] = random_person_points[:,1] - y0

            label_map = [
                [1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31],
                [4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
            ]
            temp = random_person_points[label_map[0],:]
            random_person_points[label_map[0],:] = random_person_points[label_map[1],:]
            random_person_points[label_map[1],:] = temp

            person_image = image[y0:y1,x0:x1].copy()
            person_h,person_w = person_image.shape[:2]
            person_mask = np.zeros((person_h, person_w), dtype=np.uint8)

            random_person_points = np.expand_dims(random_person_points, 0)
            vis = random_person_points[:, :, 2] > 0.02
            return {
                'image': person_image,
                'segments': person_mask,
                'joints2d': random_person_points[:,:,:2].copy(),
                'joints_vis': vis.astype(np.int32),
                'has_joints': True,
                'has_segments': False,
                'bboxes': np.array([[0,0,person_w,person_h]]),
            }


skeleton = [
    [8,6],
    [6,5],
    [5,4],
    [4,0],
    [0,1],
    [1,2],
    [2,3],
    [3,7],
    [0,9],
    [0,10],
    [9,10],
    [11,12],
    [11,13],
    [12,14],
    [11,23],
    [12,24],
    [23,24],
    [23,25],
    [24,26],
    [25,27],
    [26,28],
    [27,29],
    [27,31],
    [31,29],
    [28,32],
    [28,30],
    [30,32],
    [13,15],
    [14,16],
    [15,21],
    [15,19],
    [15,17],
    [17,19],
    [16,22],
    [16,18],
    [16,20],
    [18,20]
]

@reader.register
class PersonBaiduDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        self.joints_num = 33
        self.anno_big_file = os.path.join(self.dir,'train_with_facebox_facekeypoint_zhuohua_handbox_update4mediapipeOrder_230818.json')
        with open(self.anno_big_file, 'r') as fp:
            self.anno_big_info = json.load(fp)

        # 阔边
        self.ext_size = 30

        # DEBUG使用
        info_filter = []
        for info in self.anno_big_info:
            if info['dataset'] == 'zhuohua':
                info_filter.append(info)

        self.anno_big_info = info_filter
        print(f'baidu dataset number {len(self.anno_big_info)}')

    @property
    def size(self):
        # 返回数据集大小
        return len(self.anno_big_info)

    def sample(self, id):
        # 根据id，返回对应样本
        anno_info = self.anno_big_info[id]

        image_id = anno_info['image_id']
        image_dataset = anno_info['dataset']
        image_path = os.path.join(self.dir, 'images', image_dataset, image_id)
        # if image_dataset.startswith('mpii'):
        #     if np.random.random() < 0.5 and os.path.exists(os.path.join(self.dir, 'images', 'mpiiext', image_id)):
        #         image_path = os.path.join(self.dir, 'images', 'mpiiext', image_id)
        # if image_dataset.startswith('coco'):
        #     _, subfolder = image_dataset.split('/')
        #     if np.random.random() < 0.5 and os.path.exists(os.path.join(self.dir, 'images', 'cocoext', subfolder, image_id)):
        #         image_path = os.path.join(self.dir, 'images', 'cocoext', subfolder, image_id)

        image = cv2.imread(image_path)
        image_h, image_w = image.shape[:2]
        person_num = len(anno_info['keypoint_annotations'])
        random_person_i = np.random.choice(person_num)
        keys = list(anno_info['keypoint_annotations'].keys())
        random_person_key = keys[random_person_i]
        random_person_points = anno_info['keypoint_annotations'][random_person_key]
        random_person_bbox = anno_info['human_annotations'][random_person_key]
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = random_person_bbox
        if bbox_x0 > bbox_x1:
            t = bbox_x0
            bbox_x0 = bbox_x1
            bbox_x1 = t

        if bbox_y0 > bbox_y1:
            t = bbox_y0
            bbox_y0 = bbox_y1
            bbox_y1 = t

        random_ext_size = int(np.random.randint(5, self.ext_size))
        bbox_x0 = max(bbox_x0-random_ext_size, 0)
        bbox_y0 = max(bbox_y0-random_ext_size, 0)
        bbox_x1 = min(bbox_x1+random_ext_size, image_w)
        bbox_y1 = min(bbox_y1+random_ext_size, image_h)

        x0,y0,x1,y1 = int(bbox_x0), int(bbox_y0), int(bbox_x1), int(bbox_y1)
        random_person_points = np.array(random_person_points)
        random_person_points[:,0] = random_person_points[:,0] - x0
        random_person_points[:,1] = random_person_points[:,1] - y0

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_mask = np.ones((person_h, person_w), dtype=np.uint8) * 255

        random_person_points = np.expand_dims(random_person_points, 0)
        flag = random_person_points[:, :, 2]
        invalid_pos = np.where(flag > 1.1)
        flag[invalid_pos] = 0
        vis = flag > 0.01

        # create skeleton on person_mask
        # 仅骨架位置，有分割结果
        # for s,e in skeleton:
        #     if vis[0,s] == False or vis[0,e] == False:
        #         continue
        #     start_x,start_y = random_person_points[0,s,:2]
        #     end_x, end_y = random_person_points[0,e,:2]
        #     cv2.line(person_mask, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1, 5)

        return {
            'image': person_image,
            'segments': person_mask,
            'joints2d': random_person_points[:,:,:2].astype(np.float32).copy(),
            'joints_vis': vis.astype(np.int32),
            'has_joints': True,
            'has_segments': False,
            'has_weak_joints': False,
            'bboxes': np.array([[0,0,person_w,person_h]]),
        }


@reader.register
class PersonTikTokAndAHPDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        self.joints_num = 33
        self.tiktok_info = []
        tiktok_folder = os.path.join(self.dir, 'TikTok_dataset', 'TikTok_dataset')
        print(f'tiktok folder {tiktok_folder}')
        for subfolder in os.listdir(tiktok_folder):
            subfolder_path = os.path.join(tiktok_folder, subfolder)
            image_folder = os.path.join(subfolder_path, 'images')
            mask_folder = os.path.join(subfolder_path, 'masks')
            for filename in os.listdir(image_folder):
                self.tiktok_info.append({
                    'image': os.path.join(image_folder, filename),
                    'mask': os.path.join(mask_folder, filename)
                })
        self.tiktok_num = len(self.tiktok_info)
        print(f'tiktok num {self.tiktok_num}')

        baidu_folder = os.path.join(self.dir, 'humanseg_new', 'refined_annotation')
        self.baidu_info = []
        for line in open(os.path.join(baidu_folder, 'filter_refine_train.txt')):
            line = line.strip()
            image_file, mask_file = line.split(' ')
            if image_file.startswith('hands_2w'):
                continue

            image_file = os.path.join(baidu_folder, image_file)
            mask_file = os.path.join(baidu_folder, mask_file)
            if not os.path.exists(image_file) or not os.path.exists(mask_file):
                continue

            self.baidu_info.append(
                {
                    'image': image_file,
                    'mask': mask_file
                }
            )
        self.baidu_num = len(self.baidu_info)
        print(f'baidu num {self.baidu_num}')

        ahp_folder = os.path.join(self.dir, 'PoseSeg-AHP')
        self.ahp_anno_file = os.path.join(ahp_folder, 'poseseg_ahp_train.json')
        self.ahp_anno_info = []
        with open(self.ahp_anno_file, 'r') as fp:
            content = json.load(fp)
            self.ahp_anno_info.extend(content)
        self.ahp_num = len(self.ahp_anno_info)
        print(f'ahp num {self.ahp_num}')

        # 阔边
        self.ext_size = 40

    @property
    def size(self):
        # 返回数据集大小
        return self.ahp_num + self.tiktok_num + self.baidu_num

    def sample(self, id):
        if id < self.ahp_num:
            anno_info = self.ahp_anno_info[id]
            image = cv2.imread(os.path.join(self.dir, anno_info['image']))
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(os.path.join(self.dir, anno_info['mask']), cv2.IMREAD_GRAYSCALE)
            mask = mask/255
            mask = mask.astype(np.uint8)

            person_pos = np.where(mask == 1)
            x0 = np.min(person_pos[1])
            y0 = np.min(person_pos[0])

            x1 = np.max(person_pos[1])
            y1 = np.max(person_pos[0])

            random_ext_size = int(np.random.randint(5, self.ext_size))
            x0 = int(max(0, x0-random_ext_size))
            y0 = int(max(0, y0-random_ext_size))
            x1 = int(min(x1+random_ext_size, image_w))
            y1 = int(min(y1+random_ext_size, image_h))

            person_image = image[y0:y1,x0:x1].copy()
            person_h,person_w = person_image.shape[:2]
            person_mask = mask[y0:y1,x0:x1].copy()

            pose = np.zeros((1, self.joints_num, 3), dtype=np.float32)
            return {
                'image': person_image,
                'segments': person_mask,
                'joints2d': pose[:,:,:2].copy(),
                'joints_vis': pose[:, :,2].astype(np.int32),
                'has_joints': False,
                'has_weak_joints': True,
                'has_segments': True,
                'bboxes': np.array([[0,0,person_w,person_h]])
            }
        elif id < self.ahp_num + self.tiktok_num:
            id = id - self.ahp_num
            anno_info = self.tiktok_info[id]
            image = cv2.imread(anno_info['image'])
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(anno_info['mask'], cv2.IMREAD_GRAYSCALE)
            mask = mask/255
            mask = mask.astype(np.uint8)

            person_pos = np.where(mask == 1)
            x0 = np.min(person_pos[1])
            y0 = np.min(person_pos[0])

            x1 = np.max(person_pos[1])
            y1 = np.max(person_pos[0])

            random_ext_size = int(np.random.randint(5, self.ext_size))
            x0 = int(max(0, x0-random_ext_size))
            y0 = int(max(0, y0-random_ext_size))
            x1 = int(min(x1+random_ext_size, image_w))
            y1 = int(min(y1+random_ext_size, image_h))

            person_image = image[y0:y1,x0:x1].copy()
            person_h,person_w = person_image.shape[:2]
            person_mask = mask[y0:y1,x0:x1].copy()

            pose = np.zeros((1, self.joints_num, 3), dtype=np.float32)
            return {
                'image': person_image,
                'segments': person_mask,
                'joints2d': pose[:,:,:2].copy(),
                'joints_vis': pose[:,:,2].astype(np.int32),
                'has_joints': False,
                'has_weak_joints': True,
                'has_segments': True,
                'bboxes': np.array([[0,0,person_w,person_h]])
            }
        else:
            id = id - self.ahp_num - self.tiktok_num
            anno_info = self.baidu_info[id]
            image = cv2.imread(anno_info['image'])
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(anno_info['mask'], cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.uint8)

            person_pos = np.where(mask == 1)
            x0 = np.min(person_pos[1])
            y0 = np.min(person_pos[0])

            x1 = np.max(person_pos[1])
            y1 = np.max(person_pos[0])

            random_ext_size = int(np.random.randint(5, self.ext_size))
            x0 = int(max(0, x0-random_ext_size))
            y0 = int(max(0, y0-random_ext_size))
            x1 = int(min(x1+random_ext_size, image_w))
            y1 = int(min(y1+random_ext_size, image_h))

            person_image = image[y0:y1,x0:x1].copy()
            person_h,person_w = person_image.shape[:2]
            person_mask = mask[y0:y1,x0:x1].copy()

            pose = np.zeros((1, self.joints_num, 3), dtype=np.float32)
            return {
                'image': person_image,
                'segments': person_mask,
                'joints2d': pose[:,:,:2].copy(),
                'joints_vis': pose[:,:,2].astype(np.int32),
                'has_joints': False,
                'has_weak_joints': True,
                'has_segments': True,
                'bboxes': np.array([[0,0,person_w,person_h]])
            }


@reader.register
class PoseSeg_AHP(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        ahp_folder = os.path.join(self.dir, 'PoseSeg-AHP')
        if train_or_test == 'train':
            self.ahp_anno_file = os.path.join(ahp_folder, 'poseseg_ahp_train.json')
        else:
            self.ahp_anno_file = os.path.join(ahp_folder, 'poseseg_ahp_val.json')

        self.ahp_anno_info = []
        with open(self.ahp_anno_file, 'r') as fp:
            content = json.load(fp)
            self.ahp_anno_info.extend(content)
        self.ahp_num = len(self.ahp_anno_info)

    @property
    def size(self):
        return self.ahp_num

    def sample(self, id):
        anno_info = self.ahp_anno_info[id]
        image = cv2.imread(os.path.join(self.dir, anno_info['image']))
        image_h, image_w = image.shape[:2]
        mask = cv2.imread(os.path.join(self.dir, anno_info['mask']), cv2.IMREAD_GRAYSCALE)
        mask = mask/255
        mask = mask.astype(np.uint8)

        person_pos = np.where(mask == 1)
        x0 = np.min(person_pos[1])
        y0 = np.min(person_pos[0])

        x1 = np.max(person_pos[1])
        y1 = np.max(person_pos[0])

        random_ext_size = 0
        if self.train_or_test == 'train':
            random_ext_size = int(np.random.randint(5, self.ext_size))
        
        x0 = int(max(0, x0-random_ext_size))
        y0 = int(max(0, y0-random_ext_size))
        x1 = int(min(x1+random_ext_size, image_w))
        y1 = int(min(y1+random_ext_size, image_h))

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_mask = mask[y0:y1,x0:x1].copy()

        pose = np.zeros((1, 33, 3), dtype=np.float32)
        return {
            'image': person_image,
            'segments': person_mask,
            'joints2d': pose[:,:,:2].copy(),
            'joints_vis': pose[:, :,2].astype(np.int32),
            'has_joints': False,
            'has_weak_joints': True,
            'has_segments': True,
            'bboxes': np.array([[0,0,person_w,person_h]])
        }


@reader.register
class PersonBaiduBetaDataset(Dataset):
    def __init__(self, train_or_test="test", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        self.joints_num = 33
        self.anno_big_file = os.path.join(self.dir, 'baidu-beta.json')
        with open(self.anno_big_file, 'r') as fp:
            self.anno_big_info = json.load(fp)

        # 阔边
        self.ext_size = 50

    @property
    def size(self):
        # 返回数据集大小
        return len(self.anno_big_info)

    def sample(self, id):
        # 根据id，返回对应样本
        anno_info = self.anno_big_info[id]

        image_id = anno_info['image_id']
        image_dataset = anno_info['dataset']
        image_path = os.path.join(self.dir, image_dataset, f'{image_id.split(".")[0]}.png')

        image = cv2.imread(image_path)
        image_h, image_w = image.shape[:2]
        person_num = len(anno_info['keypoint_annotations'])
        random_person_i = np.random.choice(person_num)
        keys = list(anno_info['keypoint_annotations'].keys())
        random_person_key = keys[random_person_i]
        random_person_points = anno_info['keypoint_annotations'][random_person_key]
        random_person_bbox = anno_info['human_annotations'][random_person_key]
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = random_person_bbox
        if bbox_x0 > bbox_x1:
            t = bbox_x0
            bbox_x0 = bbox_x1
            bbox_x1 = t

        if bbox_y0 > bbox_y1:
            t = bbox_y0
            bbox_y0 = bbox_y1
            bbox_y1 = t

        random_ext_size = 20
        bbox_x0 = max(bbox_x0-random_ext_size, 0)
        bbox_y0 = max(bbox_y0-random_ext_size, 0)
        bbox_x1 = min(bbox_x1+random_ext_size, image_w)
        bbox_y1 = min(bbox_y1+random_ext_size, image_h)

        x0,y0,x1,y1 = int(bbox_x0), int(bbox_y0), int(bbox_x1), int(bbox_y1)
        random_person_points = np.array(random_person_points)
        random_person_points[:,0] = random_person_points[:,0] - x0
        random_person_points[:,1] = random_person_points[:,1] - y0

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_mask = np.ones((person_h, person_w), dtype=np.uint8) * 255

        random_person_points = np.expand_dims(random_person_points, 0)
        flag = random_person_points[:, :, 2]
        invalid_pos = np.where(flag > 1.1)
        flag[invalid_pos] = 0
        vis = flag > 0.01

        # create skeleton on person_mask
        # 仅骨架位置，有分割结果
        for s,e in skeleton:
            if vis[0,s] == False or vis[0,e] == False:
                continue
            start_x,start_y = random_person_points[0,s,:2]
            end_x, end_y = random_person_points[0,e,:2]
            cv2.line(person_mask, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1, 5)

        return {
            'image': person_image,
            'segments': person_mask,
            'joints2d': random_person_points[:,:,:2].astype(np.float32).copy(),
            'joints_vis': vis.astype(np.int32),
            'has_joints': True,
            'has_segments': False,
            'has_weak_joints': False,
            'bboxes': np.array([[0,0,person_w,person_h]]),
        }



@reader.register
class NoPoseDataset(Dataset):
    def __init__(self, train_or_test="test", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        self.joints_num = 33
        self.anno_big_file = os.path.join(self.dir, 'noposedata_ann.json')
        with open(self.anno_big_file, 'r') as fp:
            self.anno_big_info = json.load(fp)

    @property
    def size(self):
        # 返回数据集大小
        return len(self.anno_big_info)

    def sample(self, id):
        # 根据id，返回对应样本
        anno_info = self.anno_big_info[id]
        image_file = os.path.join(self.dir, anno_info['image_file'])
        image = cv2.imread(image_file)

        image_h, image_w = image.shape[:2]

        data_list = []
        for bbox in anno_info['bboxes']:
            x1,y1,x2,y2 = bbox
            keypoints_visible = np.ones((1,33,2), dtype=np.float32)
            keypoints_visible[:, :,-1] = 0

            keypoints = np.ones((1,33,2), dtype=np.float32)
            keypoints[:,:,0] = (x1+x2)/2.0
            keypoints[:,:,1] = np.linspace(y1, y2, 33)
            data_info = {
                'bbox': np.array([[x1,y1,x2,y2]], dtype=np.float32),
                'bbox_score': np.ones(1, dtype=np.float32),
                'area': np.array((x2-x1)*(y2-y1)).reshape(1).astype(np.float64),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'category_id': np.array([1]),
            }
            data_list.append(data_info)

        data = {
            'image': image
        }
        key_list = ['bbox', 'bbox_score', 'area', 'keypoints', 'keypoints_visible', 'category_id']
        for key in key_list:
            data[key] = np.concatenate([d[key] for d in data_list])

        return data


if __name__ == "__main__":
    aa = PersonBaiduDataset('train', '/workspace/dataset/mm')
    num = len(aa)
    for i in range(num):
        data = aa[i]
        print(data)