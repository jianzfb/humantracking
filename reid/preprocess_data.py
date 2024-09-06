import os
import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
from antgo.pipeline.extent import op
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
import cv2
import numpy as np

def keepratio_func(image):
    image_new = image
    rhi, rwi = image_new.shape[:2]
    aspect_ratio = 0.5
    nhi, nwi = rhi, rwi
    if abs(rwi / rhi - aspect_ratio) > 0.0001:
        if rwi / rhi > aspect_ratio:
            nwi = rwi
            nhi = int(rwi / aspect_ratio)
        else:
            nhi = rhi
            nwi = int(rhi * aspect_ratio)

    new_image = np.zeros((nhi, nwi,3), dtype=np.uint8)

    offset_x = (nwi - image.shape[1]) // 2
    new_image[:rhi, offset_x:rwi+offset_x] = image

    resized_new_image = cv2.resize(new_image, (256,512))
    return resized_new_image

def save(image_path, image):
    ff = image_path.split('/')
    print(image_path)
    person_id = ff[-2]
    person_filename = ff[-1]

    if not os.path.exists(f'/workspace/dataset/Beta-hq/hq/beta-{person_id}/'):
        os.makedirs(f'/workspace/dataset/Beta-hq/hq/beta-{person_id}/')
    cv2.imwrite(f'/workspace/dataset/Beta-hq/hq/beta-{person_id}/{person_filename}', image)


glob['image_path']('/workspace/dataset/Beta-hq/train/*/*.png'). \
    image_decode['image_path', 'image'](). \
    runas_op['image', 'nodistort_image'](func=keepratio_func). \
    runas_op[('image_path', 'nodistort_image'), 'out'](func=save).run()