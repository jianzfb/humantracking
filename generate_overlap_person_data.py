import os
import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/workspace/antgo')
from antgo.dataflow.datasynth import *
from antgo.dataflow.dataset import *
from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *


# 使用分割模型，将人分离出来，背景贴干扰人，然后分离出来的人再贴回去
folder = '/workspace/dataset'
# anno_big_file = os.path.join(folder, 'train_with_facebox_facekeypoint_zhuohua_handbox_update4mediapipeOrder_230818.json')
# anno_big_info = None
# with open(anno_big_file, 'r') as fp:
#     anno_big_info = json.load(fp)

# # 仅抽取mpii的数据
# filter_anno_big_info = []
# for info in anno_big_info:
#     image_dataset = info['dataset']
#     if image_dataset.startswith('coco'):
#         filter_anno_big_info.append(info)

# with open('./filer_coco_mediapipe.json', 'w') as fp:
#     json.dump(filter_ anno_big_info, fp)

def extract_file_path(anno_info):
    global folder
    image_id = anno_info['image_id']
    image_dataset = anno_info['dataset']
    image_path = os.path.join(folder, 'images', image_dataset, image_id)    
    return image_path


def select_person(image, anno_info):
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

    random_ext_size = 10
    bbox_x0 = max(bbox_x0-random_ext_size, 0)
    bbox_y0 = max(bbox_y0-random_ext_size, 0)
    bbox_x1 = min(bbox_x1+random_ext_size, image_w)
    bbox_y1 = min(bbox_y1+random_ext_size, image_h)

    x0,y0,x1,y1 = int(bbox_x0), int(bbox_y0), int(bbox_x1), int(bbox_y1)
    person_bbox = [x0, y0, x1, y1, random_person_i]
    person_image = image[y0:y1,x0:x1].copy()
    return person_bbox, person_image


background_person_folder = '/workspace/dataset/PoseSeg-AHP'
background_person_image_folder = os.path.join(background_person_folder, 'images')
background_person_mask_folder = os.path.join(background_person_folder, 'masks')
image_name_list = []
for file_name in os.listdir(background_person_image_folder):
    if file_name[0] == '.':
        continue
    image_name_list.append(file_name)


def generate_image(anno_info,image, person_image, person_bbox, keep_ratio_image_for_pose, offset_xy, seg):
    image_cp = image.copy()
    person_x0, person_y0, person_x1, person_y1, _ = person_bbox
    image_h, image_w = image.shape[:2]

    paste_x0 = max(person_x0 - 70, 0)
    paste_y0 = max(person_y0 - 70, 0)
    paste_x1 = min(person_x1 + 70, image_w)
    paste_y1 = min(person_y1 + 70, image_h)

    keep_ratio_image_h, keep_ratio_image_w = keep_ratio_image_for_pose.shape[:2]
    seg = seg[0,1]
    seg_h, seg_w = seg.shape[:2]

    offset_x, offset_y, offset_w, offset_h = offset_xy
    seg = cv2.resize(seg, (keep_ratio_image_w, keep_ratio_image_h))
    person_seg = seg[offset_y: offset_y+offset_h, offset_x: offset_x+offset_w].copy()
    person_seg = np.expand_dims(person_seg, -1)
    person_image_on_seg = person_image * person_seg
    inv_person_image_on_seg = person_image * person_seg
    person_h, person_w = person_image_on_seg.shape[:2]

    random_background_image_file_name = np.random.choice(image_name_list)
    random_background_image = os.path.join(background_person_image_folder, random_background_image_file_name)
    random_background_mask = os.path.join(background_person_mask_folder, random_background_image_file_name)

    random_background_image = cv2.imread(random_background_image)
    random_background_mask = cv2.imread(random_background_mask, cv2.IMREAD_GRAYSCALE)

    random_background_mask = random_background_mask/255
    random_background_mask = random_background_mask.astype(np.uint8)

    person_pos = np.where(random_background_mask == 1)
    x0 = np.min(person_pos[1])
    y0 = np.min(person_pos[0])

    x1 = np.max(person_pos[1])
    y1 = np.max(person_pos[0])

    random_background_person_image = random_background_image[y0:y1, x0:x1]
    random_background_person_mask = random_background_mask[y0:y1, x0:x1]
    random_background_h, random_background_w = random_background_person_image.shape[:2]

    if random_background_h > person_h * 0.8:
        scale = (person_h * 0.8) / random_background_h
        scaled_h, scaled_w = random_background_h * scale, random_background_w * scale
        
        scaled_h, scaled_w = int(scaled_h), int(scaled_w)
        random_background_person_image = cv2.resize(random_background_person_image, (scaled_w, scaled_h))
        random_background_person_mask = cv2.resize(random_background_person_mask, (scaled_w, scaled_h))

    if random_background_h < person_h *0.5:
        scale = (person_h * 0.8) / random_background_h
        scaled_h, scaled_w = random_background_h * scale, random_background_w * scale

        scaled_h, scaled_w = int(scaled_h), int(scaled_w)
        random_background_person_image = cv2.resize(random_background_person_image, (scaled_w, scaled_h))
        random_background_person_mask = cv2.resize(random_background_person_mask, (scaled_w, scaled_h))

    random_background_h, random_background_w = random_background_person_image.shape[:2]
    random_background_person_mask = np.expand_dims(random_background_person_mask, -1)

    random_paste_x0 = paste_x0 + np.random.randint(0, max(paste_x1-paste_x0-random_background_w, 1))
    random_paste_y0 = paste_y0 + np.random.randint(0, max(paste_y1-paste_y0-random_background_h, 1))

    random_paste_x0, random_paste_y0 = int(random_paste_x0), int(random_paste_y0)
    
    tmp_h = min(random_paste_y0+random_background_h, image_h) - random_paste_y0
    tmp_w = min(random_paste_x0+random_background_w, image_w) - random_paste_x0
    image_cp[random_paste_y0:min(random_paste_y0+random_background_h, image_h), random_paste_x0:min(random_paste_x0+random_background_w, image_w)] = \
        image_cp[random_paste_y0:min(random_paste_y0+random_background_h, image_h), random_paste_x0:min(random_paste_x0+random_background_w, image_w)] * (1.0-random_background_person_mask[:tmp_h,:tmp_w]) + \
        random_background_person_image[:tmp_h,:tmp_w] * random_background_person_mask[:tmp_h,:tmp_w]

    image_cp[person_y0:person_y1, person_x0:person_x1] = \
        image[person_y0:person_y1, person_x0:person_x1] * person_seg + \
        image_cp[person_y0:person_y1, person_x0:person_x1] * (1-person_seg)

    image_id = anno_info['image_id']
    dataset = anno_info['dataset']
    _, subfolder = dataset.split('/')
    cv2.imwrite(f'/workspace/dataset/cocoext/{subfolder}/{image_id}', image_cp)
    print('sdf')


json_dc['info']('./filer_coco_mediapipe.json'). \
    runas_op['info', 'file_path'](func=extract_file_path). \
    image_decode['file_path', 'image'](). \
    runas_op[('image', 'info'), ('person_bbox', 'person_image')](func=select_person). \
    keep_ratio_op['person_image', ('keep_ratio_image_for_pose', 'offset_xy')](aspect_ratio=1.0). \
    resize_op['keep_ratio_image_for_pose', 'resized_image_for_pose'](out_size=(256,256)). \
    inference_onnx_op['resized_image_for_pose', ('heatmap', 'offset', 'seg')](
        onnx_path='/workspace/humantracking/poseseg-epoch_66-model.onnx',
        mean=[0.491400*255, 0.482158*255, 0.4465231*255],
        std=[0.247032*255, 0.243485*255, 0.2615877*255],
        device_id=0
    ). \
    runas_op[('info', 'image', 'person_image', 'person_bbox', 'keep_ratio_image_for_pose', 'offset_xy', 'seg'), 'out'](func=generate_image).run()
