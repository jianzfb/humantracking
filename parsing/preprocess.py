from antgo.dataflow.imgaug import *
from antgo.framework.helper.dataset.builder import PIPELINES
import numpy as np
import cv2


# @PIPELINES.register_module()
# class YourProcess(object):
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         # sample is dict
#         image = sample['image']
#         pass


@PIPELINES.register_module()
class KeepRatioCenter(object):
    def __init__(self, aspect_ratio=1):
        self.aspect_ratio = aspect_ratio
        if not isinstance(self.aspect_ratio, float):
            raise TypeError("{}: input type is invalid.".format(self.aspect_ratio))

    def _random_crop_or_padding_image(self, sample):
        im = sample['image']
        height, width = im.shape[:2]
        cur_ratio = width / height

        min_x = 0
        min_y = 0
        max_x = width
        max_y = height
        rwi = max_x - min_x
        rhi = max_y - min_y
        left, right, top, bottom = min_x, max_x, min_y, max_y

        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)
        im = im[top:bottom, left:right, :].copy()
        rhi, rwi = im.shape[:2]
        height, width = im.shape[:2]

        if 'segments' in sample and sample['segments'].size != 0:
            sample['segments'] = sample['segments'][top:bottom, left:right].copy()

        # 随机填充，保持比例
        image_new = im
        rhi, rwi = image_new.shape[:2]
        if abs(rwi / rhi - self.aspect_ratio) > 0.0001:
            if rwi / rhi > self.aspect_ratio:
                nwi = rwi
                nhi = int(rwi / self.aspect_ratio)
            else:
                nhi = rhi
                nwi = int(rhi * self.aspect_ratio)

            # 随机填充
            top_padding = 0
            bottom_padding = nhi - rhi
            if nhi > rhi:
                top_padding = (nhi - rhi) // 2
                bottom_padding = (nhi - rhi) - top_padding

            left_padding = 0
            right_padding = nwi - rwi
            if nwi > rwi:
                left_padding = (nwi - rwi) // 2
                right_padding = (nwi - rwi) - left_padding

            # 调整image
            image_new = cv2.copyMakeBorder(
                image_new, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))

            # 调整segments
            if 'segments' in sample and sample['segments'].size != 0:
                # 无效位置填充255
                sample['segments'] = cv2.copyMakeBorder(
                sample['segments'], top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=255)

        sample['image'] = image_new
        sample['height'] = image_new.shape[0]
        sample['width'] = image_new.shape[1]

        if 'image_meta' in sample:
            sample['image_meta']['image_shape'] = (image_new.shape[0], image_new.shape[1])
        return sample

    def __call__(self, sample):
        im = sample['image']
        height, width = im.shape[:2]
        cur_ratio = width / height
        if abs(cur_ratio - self.aspect_ratio) > 0.000001:
            sample = self._random_crop_or_padding_image(sample)

        return sample


