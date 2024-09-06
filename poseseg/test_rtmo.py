import enum
import sys
sys.path.insert(0,'/workspace/antgo')
from models.codecs.simcc_label import SimCCLabel

from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
import cv2
import numpy as np
from antgo.pipeline.extent import op
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *

skeleton = [
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
    [14,16]
]

def debug_show(image, cls_scores, bbox_preds, kpt_vis, pose_vecs):
    a = cls_scores.flatten()
    b = bbox_preds.flatten()
    c = kpt_vis.flatten()
    d = pose_vecs.flatten()

    print((a[100], a[1000]))
    print((b[100], b[1000]))
    print((c[100], c[1000]))
    print((d[100], d[1000]))
    print('sdf')

image = np.ones((384,384,3), dtype=np.uint8)
image = image * 128
image = image.astype(np.uint8)

# placeholder['image'](image). \
#     inference_onnx_op['image', ("cls_scores", "bbox_preds", "kpt_vis", "pose_vecs")](
#         onnx_path='/workspace/humantracking/poseseg-epoch_280-model.onnx', 
#         mean=[0.491400*255, 0.482158*255, 0.4465231*255],
#         std=[0.247032*255, 0.243485*255, 0.2615877*255],
#         engine='rknn',
#         engine_args={
#             'device':'rk3588',
#             'calibration-images': '/workspace/project/sports/jumprope/calibration_square',
#             'quantize': True
#         }
#     ). \
#     runas_op[('image', "cls_scores", "bbox_preds", "kpt_vis", "pose_vecs"), 'out'](func=debug_show).run()


# 测试rknn结果
# 创建rknn C++ 工程
placeholder['image'](image). \
    inference_onnx_op['image', ("cls_scores", "bbox_preds", "kpt_vis", "pose_vecs")](
        onnx_path='/workspace/humantracking/poseseg-epoch_280-model.onnx', 
        mean=[0.491400*255, 0.482158*255, 0.4465231*255],
        std=[0.247032*255, 0.243485*255, 0.2615877*255],
        engine='rknn',
        engine_args={
            'device':'rk3588',
            'quantize': False
        }). \
    build(
        platform='android/arm64-v8a',
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [
                ('cls_scores', 'EAGLEEYE_SIGNAL_TENSOR'),
                ('bbox_preds', 'EAGLEEYE_SIGNAL_TENSOR'),
                ('kpt_vis', 'EAGLEEYE_SIGNAL_TENSOR'),
                ('pose_vecs', 'EAGLEEYE_SIGNAL_TENSOR')
            ],
            'name': 'temptest',
            'git': ''
        }
    )

# 测试工程
# placeholder['image'](image). \
#     eagleeye.exe.temptest['image', ('cls_scores', 'bbox_preds', 'kpt_vis', 'pose_vecs')](). \
#     runas_op[('image', "cls_scores", "bbox_preds", "kpt_vis", "pose_vecs"), 'out'](func=debug_show).run()


print('hello')

