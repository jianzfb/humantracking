from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper import reader
from antgo.dataflow.dataset.dataset import Dataset
import numpy as np
import torch
import os
import cv2
from PIL import Image


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
class MSMTDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None, **kwargs):
        super().__init__(dir=dir)
        # query dir
        self.query_dir = os.path.join(dir, 'query')
        # gallery dir
        self.gallery_dir = os.path.join(dir, 'gallery')

        self.query_samples = []
        for sub_folder in os.listdir(self.query_dir):
            for filename in os.listdir(os.path.join(self.query_dir, sub_folder)):
                label = filename[0:4]
                camera = filename.split('_')[2][0:2]

                query_info = {
                    'tag': 0,
                    'camera': int(camera),
                    'image': os.path.join(self.query_dir, sub_folder, filename)
                }
                if label[0:2]=='-1':
                    query_info['label'] = -1
                else:
                    query_info['label'] = int(label)

                self.query_samples.append(query_info)
        self.query_num = len(self.query_samples)

        self.gallery_samples = []
        for sub_folder in os.listdir(self.gallery_dir):
            for filename in os.listdir(os.path.join(self.gallery_dir, sub_folder)):
                label = filename[0:4]
                camera = filename.split('_')[2][0:2]

                gallery_info = {
                    'tag': 1,
                    'camera': int(camera),
                    'image': os.path.join(self.gallery_dir, sub_folder, filename)
                }
                if label[0:2]=='-1':
                    gallery_info['label'] = -1
                else:
                    gallery_info['label'] = int(label)

                self.gallery_samples.append(gallery_info)
        self.gallery_num = len(self.gallery_samples)

    @property
    def size(self):
        return len(self.query_samples) + len(self.gallery_samples)
    
    def sample(self, id):
        image = None
        tag = 0
        camera = 0
        label = 0
        if id < len(self.query_samples):
            info = self.query_samples[id]
            # image = cv2.imread(info['image'])
            image = Image.open(info['image'])
            tag = info['tag']
            camera = info['camera']
            label = info['label']
        else:
            info = self.gallery_samples[id-self.query_num]
            # image = cv2.imread(info['image'])
            image = Image.open(info['image'])
            tag = info['tag']
            camera = info['camera']
            label = info['label']

        return {
            'image': image,
            'tag': tag,
            'camera': camera,
            'label': label
        }


@reader.register
class MarketDataset(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None, **kwargs):
        super().__init__(dir=dir)
        # query dir
        self.query_dir = os.path.join(dir, 'query')
        # gallery dir
        self.gallery_dir = os.path.join(dir, 'gallery')

        self.query_samples = []
        for sub_folder in os.listdir(self.query_dir):
            for filename in os.listdir(os.path.join(self.query_dir, sub_folder)):
                label = filename[0:4]
                camera = filename.split('c')[1]
                query_info = {
                    'tag': 0,
                    'camera': int(camera[0]),
                    'image': os.path.join(self.query_dir, sub_folder, filename)
                }
                if label[0:2]=='-1':
                    query_info['label'] = -1
                else:
                    query_info['label'] = int(label)

                self.query_samples.append(query_info)
        self.query_num = len(self.query_samples)

        self.gallery_samples = []
        for sub_folder in os.listdir(self.gallery_dir):
            for filename in os.listdir(os.path.join(self.gallery_dir, sub_folder)):
                label = filename[0:4]
                # camera = filename.split('_')[2][0:2]
                camera = filename.split('c')[1]
                gallery_info = {
                    'tag': 1,
                    'camera': int(camera[0]),
                    'image': os.path.join(self.gallery_dir, sub_folder, filename)
                }
                if label[0:2]=='-1':
                    gallery_info['label'] = -1
                else:
                    gallery_info['label'] = int(label)

                self.gallery_samples.append(gallery_info)
        self.gallery_num = len(self.gallery_samples)

    @property
    def size(self):
        return len(self.query_samples) + len(self.gallery_samples)
    
    def sample(self, id):
        image = None
        tag = 0
        camera = 0
        label = 0
        if id < len(self.query_samples):
            info = self.query_samples[id]
            # image = cv2.imread(info['image'])
            image = Image.open(info['image'])
            tag = info['tag']
            camera = info['camera']
            label = info['label']
        else:
            info = self.gallery_samples[id-self.query_num]
            # image = cv2.imread(info['image'])
            image = Image.open(info['image'])
            tag = info['tag']
            camera = info['camera']
            label = info['label']

        return {
            'image': image,
            'tag': tag,
            'camera': camera,
            'label': label
        }