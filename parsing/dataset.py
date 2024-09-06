from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper import reader
from antgo.dataflow.dataset.dataset import Dataset
import numpy as np
import torch
import os
import cv2
import scipy
import json


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
class PersonParsingDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)

        # LV-MHP-v1(OK), LV-MHP-v2, ATR(OK), CIHP, LIP
        # self.sub_dataset = ['LV-MHP-v1', 'LV-MHP-v2','ATR',  'LIP', 'CIHP']
        # self.sub_loader = [self.load_lv_mhp_v1, self.load_lv_mhp_v2, self.load_atr,self.load_lip,self.load_cihp]

        self.sub_dataset = ['LV-MHP-v1','ATR','LIP', 'CIHP','LV-MHP-v2']
        self.sub_loader = [self.load_lv_mhp_v1,self.load_atr,self.load_lip,self.load_cihp,self.load_lv_mhp_v2]

        # self.sub_dataset = ['CIHP']
        # self.sub_loader = [self.load_cihp]
        self.ext_size = 40

        # 考虑类别 (来自MHP定义)
        # 0:  background
        # 1:  hat
        # 2:  hair
        # 3:  sunglass
        # 4:  upper-clothes
        # 5:  skirt
        # 6:  pants
        # 7:  dress
        # 8:  belt 
        # 9:  left-shoe
        # 10: right-shoe
        # 11: face
        # 12: left-leg
        # 13: right-leg
        # 14: left-arm 
        # 15: right-arm
        # 16: bag
        # 17: scarf
        # 18: torso-skin
        self.data_list = []
        for sub_dataset, sub_loader in zip(self.sub_dataset, self.sub_loader):
            sub_loader(os.path.join(dir, sub_dataset))

    def load_lv_mhp_v1(self, folder):
        image_folder = os.path.join(folder, 'images')
        anno_folder = os.path.join(folder, 'annotations')

        anno_map = {}
        for anno_file in os.listdir(anno_folder):
            file_name, person_num, person_id = anno_file.split('.')[0].split('_')
            if f'{file_name}.jpg' not in anno_map:
                anno_map[f'{file_name}.jpg'] = []
            
            anno_map[f'{file_name}.jpg'].append(os.path.join(anno_folder, anno_file))

        for image_file, anno_list in anno_map.items():
            self.data_list.append((os.path.join(image_folder, image_file), anno_list, None))

    def load_lv_mhp_v2(self, folder):
        # label对齐到mhp_v1
        image_folder = os.path.join(folder, 'train', 'images')
        anno_folder = os.path.join(folder, 'train', 'parsing_annos')

        colorize = np.zeros((255,1),np.uint8)
        colorize[[1,2],:] = 1       # 1,2 -> 1
        colorize[3,:] = 11          # 3 -> 11
        colorize[4,:] = 2           # 4 -> 2
        colorize[5,:] = 14          # 5 -> 14
        colorize[6,:] = 15          # 6 -> 15
        colorize[7,:] = 14          # 7 -> 14
        colorize[8,:] = 15          # 8 -> 15
        colorize[[9,10],:] = 255            # 9,10 -> 255
        colorize[[11,12,13,14,15],:] = 4    # 11,12,13,14,15 -> 4
        colorize[16,:] = 18                 # 16 ->18
        colorize[[17,18],:] = 6             # 17,18 -> 6
        colorize[19,:] = 5                  # 19 -> 5
        colorize[[20,21],:] = 255           # 20,21 -> 255
        colorize[[22,24,26,28],:] = 9       # 22,24,26,28 -> 9
        colorize[[23,25,27,29],:] = 10      # 23,25,27,29 -> 10
        colorize[[30,32],:] = 12            # 30,32 -> 12
        colorize[[31,33],:] = 13            # 31,33 -> 13
        colorize[34,:] = 4                  # 34->4
        colorize[[35,36,37,38],:] = 7       # 34,35,36,37,38 -> 7 ?
        colorize[39,:] = 255                # 39 -> 255
        colorize[40,:] = 16                 # 40 -> 16
        colorize[[41,42,44,46,49,51,52,53,54,55,56],:] = 255           # 41,42,44,46,49,51,52,53,54,55,56 -> 255
        colorize[43,:] = 8                  # 43 -> 8
        colorize[45,:] = 16                 # 45 -> 16
        colorize[[47,48],:] = 3             # 47,48 -> 3
        colorize[50,:] = 17                 # 50 -> 17
        colorize[[57,58],:] = 4             # 57,58 -> 4

        anno_map = {}
        for anno_file in os.listdir(anno_folder):
            file_name, person_num, person_id = anno_file.split('.')[0].split('_')
            if f'{file_name}.jpg' not in anno_map:
                anno_map[f'{file_name}.jpg'] = []
            anno_map[f'{file_name}.jpg'].append(os.path.join(anno_folder, anno_file))

        for image_file, anno_list in anno_map.items():
            self.data_list.append((os.path.join(image_folder, image_file), anno_list, colorize))

    def load_cihp(self, folder):
        cihp_folder = os.path.join(folder, 'instance-level_human_parsing', 'Training')
        image_folder = os.path.join(cihp_folder, 'Images')
        anno_folder = os.path.join(cihp_folder, 'Category_ids_split')
        colorize = np.zeros((255,1),np.uint8)
        colorize[1,:] = 1 # 1-> 1
        colorize[2,:] = 2 # 2->2
        colorize[3,:] = 255
        colorize[4,:] = 3
        colorize[5,:] = 4
        colorize[6,:] = 7
        colorize[7,:] = 4
        colorize[8,:] = 255
        colorize[9,:] = 6
        colorize[10,:] = 18
        colorize[11,:] = 17
        colorize[12,:] = 5
        colorize[13,:] = 11
        colorize[14,:] = 14
        colorize[15,:] = 15
        colorize[16,:] = 255     # 左右腿存在错误标注
        colorize[17,:] = 255     # 左右腿存在错误标注
        colorize[18,:] = 9
        colorize[19,:] = 10

        anno_map = {}
        for anno_file in os.listdir(anno_folder):
            file_name, person_num, person_id = anno_file.split('.')[0].split('_')
            if f'{file_name}.jpg' not in anno_map:
                anno_map[f'{file_name}.jpg'] = []
            anno_map[f'{file_name}.jpg'].append(os.path.join(anno_folder, anno_file))

        for image_file, anno_list in anno_map.items():
            self.data_list.append((os.path.join(image_folder, image_file), anno_list, colorize))

        # human_ids_folder = os.path.join(cihp_folder, 'Human')
        # temp_folder = os.path.join(cihp_folder, 'Category_ids_split')
        # for image_file in os.listdir(image_folder):
        #     image_name = image_file.split('.')[0]
        #     anno_file = f'{image_name}.png'
        #     human_file = f'{image_name}.png'

        #     anno_path = os.path.join(anno_folder, anno_file)
        #     anno_img = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)

        #     human_path = os.path.join(human_ids_folder, human_file)
        #     human_img = cv2.imread(human_path, cv2.IMREAD_GRAYSCALE)
        #     image_h,image_w = human_img.shape[:2]
        #     person_ids = set(human_img.flatten().tolist())
        #     person_num = len(person_ids) - 1
        #     count = 0
        #     for person_id in person_ids:
        #         if person_id == 0:
        #             continue

        #         pos = np.where(human_img == person_id)
        #         person_mask = np.zeros((image_h,image_w), dtype=np.uint8)
        #         person_mask[pos] = 1

        #         person_mask = person_mask * anno_img

        #         cv2.imwrite(f'{temp_folder}/{image_name}_{person_num}_{count}.png', person_mask)
        #         count += 1
        # print('sdf')

    def load_lip(self, folder):
        image_folder = os.path.join(folder, 'train_images')
        anno_folder = os.path.join(folder, 'TrainVal_parsing_annotations/train_segmentations')

        colorize = np.zeros((255,1),np.uint8)
        colorize[1,:] = 1 # 1-> 1
        colorize[2,:] = 2 # 2->2
        colorize[3,:] = 255
        colorize[4,:] = 3
        colorize[5,:] = 4
        colorize[6,:] = 7
        colorize[7,:] = 4
        colorize[8,:] = 255
        colorize[9,:] = 6
        colorize[10,:] = 255
        colorize[11,:] = 17
        colorize[12,:] = 5
        colorize[13,:] = 11
        colorize[14,:] = 14
        colorize[15,:] = 15
        colorize[16,:] = 12
        colorize[17,:] = 13
        colorize[18,:] = 9
        colorize[19,:] = 10

        for anno_file in os.listdir(anno_folder):
            image_name = anno_file.split('.')[0]
            if image_name in [
                '343728_1235204','101017_2031560','102473_513693','326108_1237881',
                '488127_1256660','50686_1267121','185989_1286867','323129_1300746',
                '551733_1261929','388508_1249418','151732_1304451','262001_2017311',
                '163985_442726','172439_226024','171272_1234916','195748_1224895',
                '402705_466701','62230_567401','18224_1760965','44795_523610',
                '502857_500087','172439_1745785','381766_1230600','475103_1715866',
                '205811_1703543','382731_1317993','376719_1306508','308541_517567',
                '383733_540743','193654_1278308','145189_519817','257874_1216893',
                '53729_1223996','478198_422178','395863_1705376','171272_1277191',
                '376864_2020174','102655_1691429','185168_1305633','185168_1305633',
                '288944_1706301','401589_1746883','416258_1261435','62707_1201938',
                '51470_456887','325915_1308554','487469_1263965','353483_1206941',
                '189936_1696417','387514_1264311','353483_1295123','17449_1275303',
                '171272_1247499','262200_450426','311076_1206293','18224_1275977',
                '91387_2014310','188532_1257488','306128_1256185','515540_1239816',
                '511410_1720889','192585_1325472','66944_2030475','180087_1259417',
                '14230_1312923','117417_1303979','39768_1747063','195750_1256193',
                '135671_1209529','262200_441352','152588_1289551','489842_1297441',
                '526057_1730902','542866_1718281','413792_1231989','561382_552031','581189_2157547','81401_1266005']:
                continue

            image_path = os.path.join(image_folder, f'{image_name}.jpg')
            anno_path = os.path.join(anno_folder, anno_file)
            self.data_list.append((image_path, [anno_path], colorize))

    def load_atr(self, folder):
        image_folder = os.path.join(folder, 'JPEGImages')
        anno_folder = os.path.join(folder, 'SegmentationClassAug')

        for image_file in os.listdir(image_folder):
            image_name = image_file.split('.')[0]
            anno_file = f'{image_name}.png'
            anno_path = os.path.join(anno_folder, anno_file)
            self.data_list.append((os.path.join(image_folder, image_file), [anno_path], None))

    @property
    def size(self):
        return len(self.data_list)

    def sample(self, id):
        image_path, anno_list, colorize = self.data_list[id]
        image = cv2.imread(image_path)
        random_anno_path = np.random.choice(anno_list)
        mask = cv2.imread(random_anno_path)[:,:,2]

        if colorize is not None:
            mask_h, mask_w = mask.shape[:2]
            relabel_mask = colorize[mask.flatten()].reshape((mask_h, mask_w))
            mask = relabel_mask

        person_pos = np.where(mask > 0)
        x0 = np.min(person_pos[1])
        y0 = np.min(person_pos[0])

        x1 = np.max(person_pos[1])
        y1 = np.max(person_pos[0])

        mask_h, mask_w = mask.shape[:2]
        random_ext_size = int(np.random.randint(5, self.ext_size))
        x0 = int(max(0, x0-random_ext_size))
        random_ext_size = int(np.random.randint(5, self.ext_size))
        y0 = int(max(0, y0-random_ext_size))
        random_ext_size = int(np.random.randint(5, self.ext_size))
        x1 = int(min(x1+random_ext_size, mask_w))
        random_ext_size = int(np.random.randint(5, self.ext_size))
        y1 = int(min(y1+random_ext_size, mask_h))

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_parsing_mask = mask[y0:y1,x0:x1].copy()

        if np.random.random() < 0.5:
            random_line_num = int(np.random.randint(1,10))
            for _ in range(random_line_num):
                x0,y0 = np.random.randint(0,person_w), np.random.randint(0,person_h)
                x1,y1 = np.random.randint(0,person_w), np.random.randint(0,person_h)
                cv2.line(person_image, (x0,y0), (x1,y1), (255,255,255), 2)

        person_mask = person_parsing_mask.copy()
        person_mask[person_mask > 0] = 1

        # random rotation
        height, width = person_image.shape[:2]
        cx, cy = width // 2, height // 2
        angle = np.random.randint(0, 30 * 2) - 30
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        person_image = cv2.warpAffine(
                src=person_image,
                M=rot_mat,
                dsize=person_image.shape[1::-1],
                flags=cv2.INTER_AREA)

        person_parsing_mask = cv2.warpAffine(
                person_parsing_mask,
                M=rot_mat,
                dsize=person_parsing_mask.shape[1::-1],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255)

        person_mask = cv2.warpAffine(
                person_mask,
                M=rot_mat,
                dsize=person_mask.shape[1::-1],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255)

        # resize
        person_image = cv2.resize(person_image, (256, 256))
        person_parsing_mask = cv2.resize(person_parsing_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        person_mask = cv2.resize(person_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        return {
            'image': person_image,
            'parsing_body_segment': person_parsing_mask,
            'whole_body_segment': person_mask,
            'has_parsing_body': True
        }


@reader.register
class PersonSegDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)

        # TikTok_dataset
        # PoseSeg-AHP
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

        # 阔边
        self.ext_size = 40

    @property
    def size(self):
        # 返回数据集大小
        return self.ahp_num + self.tiktok_num
    
    def sample(self, id):
        if id < self.ahp_num:
            anno_info = self.ahp_anno_info[id]
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

            # rotation
            height, width = person_image.shape[:2]
            cx, cy = width // 2, height // 2
            angle = np.random.randint(0, 30 * 2) - 30
            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

            person_image = cv2.warpAffine(
                    src=person_image,
                    M=rot_mat,
                    dsize=person_image.shape[1::-1],
                    flags=cv2.INTER_AREA)

            person_mask = cv2.warpAffine(
                    person_mask,
                    M=rot_mat,
                    dsize=person_mask.shape[1::-1],
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255)

            # resize
            person_image = cv2.resize(person_image, (256, 256))
            person_mask = cv2.resize(person_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            return {
                'image': person_image,
                'whole_body_segment': person_mask,
                'parsing_body_segment': np.zeros(person_mask.shape, dtype=np.uint8),
                'has_parsing_body': False
            }
        else:
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

            # rotation
            height, width = person_image.shape[:2]
            cx, cy = width // 2, height // 2
            angle = np.random.randint(0, 30 * 2) - 30
            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

            person_image = cv2.warpAffine(
                    src=person_image,
                    M=rot_mat,
                    dsize=person_image.shape[1::-1],
                    flags=cv2.INTER_AREA)

            person_mask = cv2.warpAffine(
                    person_mask,
                    M=rot_mat,
                    dsize=person_mask.shape[1::-1],
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255)

            # resize
            person_image = cv2.resize(person_image, (256, 256))
            person_mask = cv2.resize(person_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            return {
                'image': person_image,
                'whole_body_segment': person_mask,
                'parsing_body_segment': np.zeros(person_mask.shape, dtype=np.uint8),
                'has_parsing_body': False
            }


@reader.register
class PersonParsingTestDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)

        # self.sub_dataset = ['CIHP','LV-MHP-v2']
        # self.sub_loader = [self.load_cihp,self.load_lv_mhp_v2]

        self.sub_dataset = ['LV-MHP-v2']
        self.sub_loader = [self.load_lv_mhp_v2]

        # 考虑类别 (来自MHP定义)
        # 0:  background
        # 1:  hat
        # 2:  hair
        # 3:  sunglass
        # 4:  upper-clothes
        # 5:  skirt
        # 6:  pants
        # 7:  dress
        # 8:  belt 
        # 9:  left-shoe
        # 10: right-shoe
        # 11: face
        # 12: left-leg
        # 13: right-leg
        # 14: left-arm 
        # 15: right-arm
        # 16: bag
        # 17: scarf
        # 18: torso-skin
        self.data_list = []
        for sub_dataset, sub_loader in zip(self.sub_dataset, self.sub_loader):
            sub_loader(os.path.join(dir, sub_dataset))

    def load_lv_mhp_v2(self, folder):
        # label对齐到mhp_v1
        image_folder = os.path.join(folder, 'val', 'images')
        anno_folder = os.path.join(folder, 'val', 'parsing_annos')

        colorize = np.zeros((255,1),np.uint8)
        colorize[[1,2],:] = 1       # 1,2 -> 1
        colorize[3,:] = 11          # 3 -> 11
        colorize[4,:] = 2           # 4 -> 2
        colorize[5,:] = 14          # 5 -> 14
        colorize[6,:] = 15          # 6 -> 15
        colorize[7,:] = 14          # 7 -> 14
        colorize[8,:] = 15          # 8 -> 15
        colorize[[9,10],:] = 255            # 9,10 -> 255
        colorize[[11,12,13,14,15],:] = 4    # 11,12,13,14,15 -> 4
        colorize[16,:] = 18                 # 16 ->18
        colorize[[17,18],:] = 6             # 17,18 -> 6
        colorize[19,:] = 5                  # 19 -> 5
        colorize[[20,21],:] = 255           # 20,21 -> 255
        colorize[[22,24,26,28],:] = 9       # 22,24,26,28 -> 9
        colorize[[23,25,27,29],:] = 10      # 23,25,27,29 -> 10
        colorize[[30,32],:] = 12            # 30,32 -> 12
        colorize[[31,33],:] = 13            # 31,33 -> 13
        colorize[34,:] = 4                  # 34->4
        colorize[[35,36,37,38],:] = 7       # 34,35,36,37,38 -> 7 ?
        colorize[39,:] = 255                # 39 -> 255
        colorize[40,:] = 16                 # 40 -> 16
        colorize[[41,42,44,46,49,51,52,53,54,55,56],:] = 255           # 41,42,44,46,49,51,52,53,54,55,56 -> 255
        colorize[43,:] = 8                  # 43 -> 8
        colorize[45,:] = 16                 # 45 -> 16
        colorize[[47,48],:] = 3             # 47,48 -> 3
        colorize[50,:] = 17                 # 50 -> 17
        colorize[[57,58],:] = 4             # 57,58 -> 4

        anno_map = {}
        for anno_file in os.listdir(anno_folder):
            file_name, person_num, person_id = anno_file.split('.')[0].split('_')
            if f'{file_name}.jpg' not in anno_map:
                anno_map[f'{file_name}.jpg'] = []
            anno_map[f'{file_name}.jpg'].append(os.path.join(anno_folder, anno_file))

        for image_file, anno_list in anno_map.items():
            self.data_list.append((os.path.join(image_folder, image_file), anno_list, colorize))

    def load_cihp(self, folder):
        cihp_folder = os.path.join(folder, 'instance-level_human_parsing', 'Validation')
        image_folder = os.path.join(cihp_folder, 'Images')
        anno_folder = os.path.join(cihp_folder, 'Category_ids_split')
        colorize = np.zeros((255,1),np.uint8)
        colorize[1,:] = 1 # 1-> 1
        colorize[2,:] = 2 # 2->2
        colorize[3,:] = 255
        colorize[4,:] = 3
        colorize[5,:] = 4
        colorize[6,:] = 7
        colorize[7,:] = 4
        colorize[8,:] = 255
        colorize[9,:] = 6
        colorize[10,:] = 18
        colorize[11,:] = 17
        colorize[12,:] = 5
        colorize[13,:] = 11
        colorize[14,:] = 14
        colorize[15,:] = 15
        colorize[16,:] = 255     # 左右腿存在错误标注
        colorize[17,:] = 255     # 左右腿存在错误标注
        colorize[18,:] = 9
        colorize[19,:] = 10

        anno_map = {}
        for anno_file in os.listdir(anno_folder):
            file_name, person_num, person_id = anno_file.split('.')[0].split('_')
            if f'{file_name}.jpg' not in anno_map:
                anno_map[f'{file_name}.jpg'] = []
            anno_map[f'{file_name}.jpg'].append(os.path.join(anno_folder, anno_file))

        for image_file, anno_list in anno_map.items():
            self.data_list.append((os.path.join(image_folder, image_file), anno_list, colorize))

        # human_ids_folder = os.path.join(cihp_folder, 'Human')
        # temp_folder = os.path.join(cihp_folder, 'Category_ids_split')
        # for image_file in os.listdir(image_folder):
        #     image_name = image_file.split('.')[0]
        #     anno_file = f'{image_name}.png'
        #     human_file = f'{image_name}.png'

        #     anno_path = os.path.join(anno_folder, anno_file)
        #     anno_img = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)

        #     human_path = os.path.join(human_ids_folder, human_file)
        #     human_img = cv2.imread(human_path, cv2.IMREAD_GRAYSCALE)
        #     image_h,image_w = human_img.shape[:2]
        #     person_ids = set(human_img.flatten().tolist())
        #     person_num = len(person_ids) - 1
        #     count = 0
        #     for person_id in person_ids:
        #         if person_id == 0:
        #             continue

        #         pos = np.where(human_img == person_id)
        #         person_mask = np.zeros((image_h,image_w), dtype=np.uint8)
        #         person_mask[pos] = 1

        #         person_mask = person_mask * anno_img

        #         cv2.imwrite(f'{temp_folder}/{image_name}_{person_num}_{count}.png', person_mask)
        #         count += 1
        # print('sdf')

    def load_lip(self, folder):
        image_folder = os.path.join(folder, 'val_images')
        anno_folder = os.path.join(folder, 'TrainVal_parsing_annotations/val_segmentations')

        colorize = np.zeros((255,1),np.uint8)
        colorize[1,:] = 1 # 1-> 1
        colorize[2,:] = 2 # 2->2
        colorize[3,:] = 255
        colorize[4,:] = 3
        colorize[5,:] = 4
        colorize[6,:] = 7
        colorize[7,:] = 4
        colorize[8,:] = 255
        colorize[9,:] = 6
        colorize[10,:] = 255
        colorize[11,:] = 17
        colorize[12,:] = 5
        colorize[13,:] = 11
        colorize[14,:] = 14
        colorize[15,:] = 15
        colorize[16,:] = 12
        colorize[17,:] = 13
        colorize[18,:] = 9
        colorize[19,:] = 10

        for anno_file in os.listdir(anno_folder):
            image_name = anno_file.split('.')[0]
            if image_name in [
                '343728_1235204','101017_2031560','102473_513693','326108_1237881',
                '488127_1256660','50686_1267121','185989_1286867','323129_1300746',
                '551733_1261929','388508_1249418','151732_1304451','262001_2017311',
                '163985_442726','172439_226024','171272_1234916','195748_1224895',
                '402705_466701','62230_567401','18224_1760965','44795_523610',
                '502857_500087','172439_1745785','381766_1230600','475103_1715866',
                '205811_1703543','382731_1317993','376719_1306508','308541_517567',
                '383733_540743','193654_1278308','145189_519817','257874_1216893',
                '53729_1223996','478198_422178','395863_1705376','171272_1277191',
                '376864_2020174','102655_1691429','185168_1305633','185168_1305633',
                '288944_1706301','401589_1746883','416258_1261435','62707_1201938',
                '51470_456887','325915_1308554','487469_1263965','353483_1206941',
                '189936_1696417','387514_1264311','353483_1295123','17449_1275303',
                '171272_1247499','262200_450426','311076_1206293','18224_1275977',
                '91387_2014310','188532_1257488','306128_1256185','515540_1239816',
                '511410_1720889','192585_1325472','66944_2030475','180087_1259417',
                '14230_1312923','117417_1303979','39768_1747063','195750_1256193',
                '135671_1209529','262200_441352','152588_1289551','489842_1297441',
                '526057_1730902','542866_1718281','413792_1231989','561382_552031','581189_2157547','81401_1266005']:
                continue

            image_path = os.path.join(image_folder, f'{image_name}.jpg')
            anno_path = os.path.join(anno_folder, anno_file)
            self.data_list.append((image_path, [anno_path], colorize))

    @property
    def size(self):
        return len(self.data_list)

    def sample(self, id):
        image_path, anno_list, colorize = self.data_list[id]
        image = cv2.imread(image_path)
        random_anno_path = np.random.choice(anno_list)
        mask = cv2.imread(random_anno_path)[:,:,2]

        if colorize is not None:
            mask_h, mask_w = mask.shape[:2]
            relabel_mask = colorize[mask.flatten()].reshape((mask_h, mask_w))
            mask = relabel_mask

        person_pos = np.where(mask > 0)
        x0 = np.min(person_pos[1])
        y0 = np.min(person_pos[0])

        x1 = np.max(person_pos[1])
        y1 = np.max(person_pos[0])

        mask_h, mask_w = mask.shape[:2]
        x0 = int(max(0, x0-10))
        y0 = int(max(0, y0-10))
        x1 = int(min(x1+10, mask_w))
        y1 = int(min(y1+10, mask_h))

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_parsing_mask = mask[y0:y1,x0:x1].copy()

        # resize
        person_image = cv2.resize(person_image, (256, 256))
        person_parsing_mask = cv2.resize(person_parsing_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        return {
            'image': person_image,
            'segments': person_parsing_mask,
        }



if __name__ == "__main__":
    pp = PersonParsingDataset(train_or_test='train',dir='/workspace/dataset/humanparsing')
    num = len(pp)
    for i in range(num):
        data = pp[i]