import os
import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/workspace/antgo')
from antgo.dataflow.datasynth import *
from antgo.dataflow.dataset import *
from antgo.utils.sample_gt import *
import mediapipe
from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
from antgo.pipeline.extent import op
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
from antgo.pipeline.functional.common.detector import *
import random
from antgo.utils.sample_gt import *

with open('/workspace/humantracking/project-2-at-2023-09-19-11-09-423d991a.json', 'r') as fp:
    sample_anno_list = json.load(fp)

gt_info_map = {}
for info in sample_anno_list:
    gt_info = {'bbox': [], 'label': []}
    for i in range(len(info['annotations'][0]['result'])):
        sample_anno_instance = info['annotations'][0]['result'][i]
        height = sample_anno_instance['original_height']
        width = sample_anno_instance['original_width']  

        bbox_x = sample_anno_instance['value']['x'] / 100.0 * width
        bbox_y = sample_anno_instance['value']['y'] / 100.0 * height
        bbox_width = sample_anno_instance['value']['width'] / 100.0 * width
        bbox_height = sample_anno_instance['value']['height'] / 100.0 * height
        bbox_label = sample_anno_instance['value']['rectanglelabels'][0]
        bbox = [bbox_x,bbox_y,bbox_x+bbox_width,bbox_y+bbox_height]
        gt_info['bbox'].append(bbox)
        if bbox_label == 'person':
            gt_info['label'].append(0)
        else:
            gt_info['label'].append(1)

    name = info['data']['image'].split('/')[-1][9:]
    if '_' in name:
        gt_info_map[name] = gt_info


# 生成过程
count = 0
version = 20
anno_list = []
sgt = SampleGTTemplate()
def sync_show(image_path, image, sync):
    global gt_info_map
    layout_image = sync['layout_image']
    layout_mask = sync['layout_mask']

    info = {'bbox': [], 'label': []}
    image_name = os.path.basename(image_path)
    if image_name == '12_26_1695110479-99.png':
        print('sdf')
    if image_name in gt_info_map:
        info = gt_info_map[image_name]
    
    pos = np.where(layout_mask == 1)
    x0 = np.min(pos[1])
    x1 = np.max(pos[1])
    y0 = np.min(pos[0])
    y1 = np.max(pos[0])
    info['bbox'].append([float(x0),float(y0),float(x1),float(y1)])
    info['label'].append(1)

    global count
    global anno_list
    # for bb, ll in zip(info['bbox'], info['label']):
    #     if int(ll) == 0:
    #         layout_image = cv2.rectangle(layout_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2)
    #     else:
    #         layout_image = cv2.rectangle(layout_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 2)
    gt_info = sgt.get()    
    gt_info['image_file'] = f'/workspace/dataset/badcase-sync-{version}/{count}.png'
    gt_info['height'] = int(layout_image.shape[0])
    gt_info['width'] = int(layout_image.shape[1])
    gt_info['bboxes'] = info['bbox']
    gt_info['labels'] = info['label']

    anno_list.append(gt_info)
    if not os.path.exists(f'/workspace/dataset/badcase-sync-{version}'):
        os.makedirs(f'/workspace/dataset/badcase-sync-{version}')
    cv2.imwrite(f'/workspace/dataset/badcase-sync-{version}/{count}.png', layout_image)
    count += 1


class LayoutGenerator:
    def __init__(self):
        self._degree = 45

    def scale(self):
        return 0.05, 0.2

    def __call__(self, image):
        ball_index = np.random.choice(8)
        image = cv2.imread(f'/workspace/dataset/bad-ball/bad-ball-{ball_index}.png', cv2.IMREAD_UNCHANGED)
        # 随机旋转
        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2
        angle = np.random.randint(0, self._degree * 2) - self._degree
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image = cv2.warpAffine(
                src=image,
                M=rot_mat,
                dsize=image.shape[1::-1],
                flags=cv2.INTER_AREA)

        return image, {}


glob['image_path']('/workspace/dataset/invlight/*'). \
    image_decode['image_path', 'image']() .\
    sync_layout_op['image', 'layout-1'](layout_gen=[LayoutGenerator()], layout_id=[1]). \
    sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.05, max_scale=0.2). \
    runas_op[('image_path','image', 'sync-out'), 'out'](func=sync_show).run()

with open(f'./badcase_sync_{version}.json', 'w') as fp:
    json.dump(anno_list, fp)

# # 清理背景图
# op.load('detpostprocessop', '/workspace/project/sports/volleyball')
# invalid_file_list = []
# def record_func(image_path, bboxes):
#     global invalid_file_list
#     if bboxes.shape[0] != 0:
#         invalid_file_list.append(image_path)
#     else:
#         print(f'skip {image_path}')

# glob['image_path']('/workspace/dataset/mm/dataset/background/*/test/*').\
#     image_decode['image_path', 'image']() .\
#     resize_op['image', 'resized_image_for_person'](out_size=(704,384)). \
#     inference_onnx_op['resized_image_for_person', 'output'](
#         mean=(128, 128, 128),
#         std=(128, 128, 128),
#         onnx_path='/workspace/humantracking/coco-epoch_40-model.onnx'
#     ). \
#     deploy.detpostprocess_func[('image','output'), ('bboxes', 'labels')](level_hw=np.array([48,88,24,44,12,22], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[('image_path', 'bboxes'), 'out'](func=record_func).run()

# with open('./invalid_file.json', 'w') as fp:
#     json.dump(invalid_file_list, fp)

# with open('/workspace/humantracking/invalid_file.json', 'r') as fp:
#     content = json.load(fp)
#     for file_path in content:
#         print(file_path)
#         if os.path.exists(file_path):
#             os.remove(file_path)