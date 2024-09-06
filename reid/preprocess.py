from antgo.dataflow.imgaug import *
from antgo.framework.helper.dataset.builder import PIPELINES
import numpy as np
import cv2
from PIL import Image


# @PIPELINES.register_module()
# class YourProcess(object):
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         # sample is dict
#         image = sample['image']
#         pass


@PIPELINES.register_module()
class KeepRatioTopAlign(object):
    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio
    
    def __call__(self, sample):
        image = np.array(sample['image'])
        rhi, rwi = image.shape[:2]
        aspect_ratio = self.aspect_ratio
        nhi, nwi = rhi, rwi
        if abs(rwi / rhi - aspect_ratio) > 0.0001:
            if rwi / rhi > aspect_ratio:
                nwi = rwi
                nhi = int(rwi / aspect_ratio)
            else:
                nhi = rhi
                nwi = int(rhi * aspect_ratio)

        new_image = np.zeros((nhi, nwi,3), dtype=np.uint8)
        
        offset_y = 0
        if np.random.random() > 0.5 and nhi > image.shape[0]:
            offset_y = int(np.random.randint(0, (nhi - image.shape[0])))

        offset_x = 0
        if np.random.random() > 0.5 and nwi > image.shape[1]:
            offset_x = int(np.random.randint(0, (nwi - image.shape[1])))
        new_image[offset_y:rhi+offset_y, offset_x:rwi+offset_x] = image

        if new_image.shape[1] > 64:
            if np.random.random() < 0.3:
                new_image = cv2.resize(new_image, (64,128))

        resized_new_image = cv2.resize(new_image, (128,256))
        sample['image'] = Image.fromarray(resized_new_image)
        return sample


@PIPELINES.register_module()
class Show(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.array(sample['image'])
        cv2.imwrite("./a.png", image)
        print('h')
        return sample