from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper import reader
from antgo.dataflow.dataset.dataset import Dataset
import numpy as np
import torch
import os
import json
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
class PersonSegDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        
        # TikTok_dataset
        # PoseSeg-AHP
        # Sudoku
        # humanseg_new
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

        ahp_folder = os.path.join(self.dir, 'PoseSeg-AHP')
        self.ahp_anno_file = os.path.join(ahp_folder, 'poseseg_ahp_train_new.json')
        self.ahp_anno_info = []
        with open(self.ahp_anno_file, 'r') as fp:
            content = json.load(fp)

            for sample in content:
                self.ahp_anno_info.append({
                    'image': os.path.join(ahp_folder, sample['image']),
                    'mask': os.path.join(ahp_folder, sample['mask']),
                })
        self.ahp_num = len(self.ahp_anno_info)
        print(f'ahp num {self.ahp_num}')

        sudoku_folder = os.path.join(self.dir, 'sudoku')
        self.sudoku_anno_file = os.path.join(sudoku_folder, 'train.json')
        self.sudoku_anno_info = []
        with open(self.sudoku_anno_file, 'r') as fp:
            content = json.load(fp)

            for sample in content:
                self.sudoku_anno_info.append({
                    'image': os.path.join(sudoku_folder, sample['image']),
                    'mask': os.path.join(sudoku_folder, sample['mask']),
                })
        self.sudoku_num = len(self.sudoku_anno_info)
        print(f'sudoku num {self.sudoku_num}')

        baidu_folder = os.path.join(self.dir, 'humanseg_new', 'refined_annotation')
        self.baidu_info = []
        # for line in open(os.path.join(baidu_folder, 'filter_refine_train.txt')):
        #     line = line.strip()
        #     image_file, mask_file = line.split(' ')
        #     if image_file.startswith('hands_2w'):
        #         continue

        #     image_file = os.path.join(baidu_folder, image_file)
        #     mask_file = os.path.join(baidu_folder, mask_file)
        #     if not os.path.exists(image_file) or not os.path.exists(mask_file):
        #         continue

        #     self.baidu_info.append(
        #         {
        #             'image': image_file,
        #             'mask': mask_file
        #         }
        #     )
        self.baidu_num = len(self.baidu_info)
        print(f'baidu num {self.baidu_num}')

        tiqianqu_folder = os.path.join(self.dir, 'tiqianqu-ext')
        self.tiqianqu_info = []
        for image_file in os.listdir(os.path.join(tiqianqu_folder, 'image')):
            if image_file[0] == '.':
                continue
            mask_file_path = os.path.join(tiqianqu_folder, 'mask')
            image_file_path = os.path.join(tiqianqu_folder, 'image')

            self.tiqianqu_info.append({
                'image': os.path.join(image_file_path, image_file),
                'mask': os.path.join(mask_file_path, image_file)
            })
        self.tiqianqu_num = len(self.tiqianqu_info)

        # 阔边
        self.ext_size = 40

    @property
    def size(self):
        # 返回数据集大小
        return self.ahp_num + self.tiktok_num + self.baidu_num + self.sudoku_num + self.tiqianqu_num
    
    def sample(self, id):
        if id < self.ahp_num:
            anno_info = self.ahp_anno_info[id]
            image = cv2.imread(anno_info['image'])
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(anno_info['mask'], cv2.IMREAD_GRAYSCALE)
            mask = mask/255
            mask = mask.astype(np.uint8)

            return {
                'image': image,
                'segments': mask,
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
            person_mask = mask[y0:y1,x0:x1].copy()
            return {
                'image': person_image,
                'segments': person_mask,
            }
        elif id < self.ahp_num + self.tiktok_num + self.baidu_num:
            id = id - self.ahp_num - self.tiktok_num
            anno_info = self.baidu_info[id]
            image = cv2.imread(anno_info['image'])
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(anno_info['mask'], cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.uint8)

            return {
                'image': image,
                'segments': mask,
            }
        elif id < self.ahp_num + self.tiktok_num + self.baidu_num + self.sudoku_num:
            id = id - self.ahp_num - self.tiktok_num - self.baidu_num
            anno_info = self.sudoku_anno_info[id]
            image = cv2.imread(anno_info['image'])
            image_h, image_w = image.shape[:2]
            mask = cv2.imread(anno_info['mask'], cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.uint8)

            return {
                'image': image,
                'segments': mask,
            }
        else:
            id = id - self.ahp_num - self.tiktok_num - self.baidu_num - self.sudoku_num
            anno_info = self.tiqianqu_info[id]
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
            person_mask = mask[y0:y1,x0:x1].copy()
            return {
                'image': person_image,
                'segments': person_mask,
            }
