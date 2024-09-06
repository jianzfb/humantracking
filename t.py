import sys
import os
import json
import numpy as np
import cv2
sys.path.insert(0, '/workspace/antgo')
from antgo.dataflow.datasynth import *
from antgo.dataflow.dataset import *
from antgo.utils.sample_gt import *

# # 创建模板图
# with open('/workspace/dataset/ext/ball-seg-anno.json', 'r') as fp:
#     sample_anno_list = json.load(fp)

# for sample_i in range(len(sample_anno_list)):
#     for index in range(len(sample_anno_list[sample_i]['annotations'][0]['result'])):
#         sample_anno_instance = sample_anno_list[sample_i]['annotations'][0]['result'][index]
#         height = sample_anno_instance['original_height']
#         width = sample_anno_instance['original_width']

#         points = sample_anno_instance['value']['points']
#         label_name = sample_anno_instance['value']['polygonlabels'][0]
#         # label_id = label_name_and_label_id_map[label_name]

#         points_array = np.array(points) 
#         points_array[:, 0] = points_array[:, 0] / 100.0 * width
#         points_array[:, 1] = points_array[:, 1] / 100.0 * height
#         points_array = points_array.astype(np.int32)
#         name = sample_anno_list[sample_i]['file_upload'].split('/')[-1][9:]

#         image = cv2.imread(f'/workspace/dataset/ext/ball-ext-dataset/images/{name}')
#         mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#         mask = cv2.fillConvexPoly(mask, points_array, 255)

#         if label_name == 'cushion':
#             name = f'cushion_{name}'
#         cv2.imwrite(f'/workspace/dataset/ext/ball-ext-dataset/imagewithmask/{name}', np.concatenate([image, np.expand_dims(mask, -1)], -1))


########################
# with open('/workspace/dataset/ext/person_ball_anno.json', 'r') as fp:
#     sample_anno_list = json.load(fp)

# crop_name = 'c30'
# anno_info_list = []
# sgt = SampleGTTemplate()
# for info in sample_anno_list:
#     # gt_info = {'bbox': [], 'label': []}
#     file_name = info['data']['image'].split('/')[-1][9:]
#     sub_folder = info['data']['image'].split('/')[-1][9:].split('_rate')[0]
#     gt_info = sgt.get()    
#     gt_info['image_file'] = os.path.join('/workspace/dataset/ext/personball-ext-dataset/image/', sub_folder, file_name)
#     image = cv2.imread(gt_info['image_file'])

#     is_big_image = False
#     if image is None:
#         is_big_image = True
#         file_name = info['data']['image'].split('/')[-1][9:]
#         gt_info['image_file'] = os.path.join('/workspace/dataset/ext/ball-ext-dataset/images/', file_name)
#         image = cv2.imread(gt_info['image_file'])

#     if not is_big_image:
#         continue

#     assert(image is not None)
#     image_h, image_w = image.shape[:2]

#     is_ok = False
#     try_count = 0
#     while try_count < 10:
#         try_count += 1

#         crop_x0 = int(np.random.randint(0, image_w//2))
#         crop_y0 = int(np.random.randint(0, image_h//2))
#         crop_x1 = crop_x0 + image_w//2
#         crop_y1 = crop_y0 + image_h//2
#         crop_image = image[crop_y0:crop_y1, crop_x0:crop_x1]

#         gt_info['bboxes'] = []
#         gt_info['labels'] = []
#         gt_info['image_file'] = os.path.join(f'/workspace/dataset/ext/ball-ext-dataset/crop_{crop_name}_images/', file_name)
#         if not os.path.exists(f'/workspace/dataset/ext/ball-ext-dataset/crop_{crop_name}_images/'):
#             os.makedirs(f'/workspace/dataset/ext/ball-ext-dataset/crop_{crop_name}_images/')

#         obj_num = len(info['annotations'][0]['result'])
#         for i in range(len(info['annotations'][0]['result'])):
#             sample_anno_instance = info['annotations'][0]['result'][i]
#             height = sample_anno_instance['original_height']
#             width = sample_anno_instance['original_width']  
#             gt_info['height'] = height
#             gt_info['width'] = width

#             bbox_x = sample_anno_instance['value']['x'] / 100.0 * width
#             bbox_y = sample_anno_instance['value']['y'] / 100.0 * height
#             bbox_width = sample_anno_instance['value']['width'] / 100.0 * width
#             bbox_height = sample_anno_instance['value']['height'] / 100.0 * height
#             bbox_label = sample_anno_instance['value']['rectanglelabels'][0]
#             if not (bbox_x >= crop_x0 and bbox_y >= crop_y0 and (bbox_x + bbox_width) < crop_x1 and (bbox_y + bbox_height) < crop_y1):
#                 continue

#             print(f'bbox_label {bbox_label}')
#             bbox_x0 = bbox_x - crop_x0
#             bbox_y0 = bbox_y - crop_y0
#             bbox_x1 = int(bbox_x+bbox_width) - crop_x0
#             bbox_y1 = int(bbox_y+bbox_height) - crop_y0

#             bbox = [bbox_x0, bbox_y0, bbox_x1, bbox_y1]
#             gt_info['bboxes'].append(bbox)
#             if bbox_label == 'person':
#                 gt_info['labels'].append(0)
#                 # cv2.rectangle(crop_image, (int(bbox_x0),int(bbox_y0)), (int(bbox_x1), int(bbox_y1)), (0,255,0), 2)
#             else:
#                 gt_info['labels'].append(1)
#                 # cv2.rectangle(crop_image, (int(bbox_x0),int(bbox_y0)), (int(bbox_x1), int(bbox_y1)), (0,0,255), 2)

#         if len(gt_info['bboxes']) != obj_num:
#             continue

#         cv2.imwrite(gt_info['image_file'], crop_image)
#         is_ok = True
#         break

#     if not is_ok:
#         # 无效，掠过
#         continue
#     anno_info_list.append(gt_info)

# with open(f'/workspace/dataset/ext/ball-ext-dataset/ext_crop_{crop_name}_zhanhui_ball.json', 'w') as fp:
#     json.dump(anno_info_list, fp)
# print('sdf')

#############################
with open('/workspace/humantracking/zhanhuireal2/zhanhuireal2.json', 'r') as fp:
    sample_anno_list = json.load(fp)

anno_info_list = []
sgt = SampleGTTemplate()
for info in sample_anno_list:
    # gt_info = {'bbox': [], 'label': []}
    file_name = info['data']['image'].split('/')[-1][9:]
    # sub_folder = info['data']['image'].split('/')[-1][9:].split('_rate')[0]
    sub_folder='use'
    gt_info = sgt.get()    
    gt_info['image_file'] = os.path.join('/workspace/humantracking/zhanhuireal2', sub_folder, file_name)
    image = cv2.imread(gt_info['image_file'])

    # is_big_image = False
    # if image is None:
    #     is_big_image = True
    #     file_name = info['data']['image'].split('/')[-1][9:]
    #     gt_info['image_file'] = os.path.join('/workspace/dataset/ext/ball-ext-dataset/images/', file_name)
    #     image = cv2.imread(gt_info['image_file'])

    assert(image is not None)
    image_h, image_w = image.shape[:2]

    obj_num = len(info['annotations'][0]['result'])
    for i in range(len(info['annotations'][0]['result'])):
        sample_anno_instance = info['annotations'][0]['result'][i]
        height = sample_anno_instance['original_height']
        width = sample_anno_instance['original_width']  
        gt_info['height'] = height
        gt_info['width'] = width

        bbox_x = sample_anno_instance['value']['x'] / 100.0 * width
        bbox_y = sample_anno_instance['value']['y'] / 100.0 * height
        bbox_width = sample_anno_instance['value']['width'] / 100.0 * width
        bbox_height = sample_anno_instance['value']['height'] / 100.0 * height
        bbox_label = sample_anno_instance['value']['rectanglelabels'][0]

        print(f'bbox_label {bbox_label}')
        bbox_x0 = bbox_x
        bbox_y0 = bbox_y
        bbox_x1 = int(bbox_x+bbox_width)
        bbox_y1 = int(bbox_y+bbox_height)

        bbox = [bbox_x0, bbox_y0, bbox_x1, bbox_y1]
        gt_info['bboxes'].append(bbox)
        if bbox_label == 'person':
            gt_info['labels'].append(0)
            # cv2.rectangle(crop_image, (int(bbox_x0),int(bbox_y0)), (int(bbox_x1), int(bbox_y1)), (0,255,0), 2)
        else:
            gt_info['labels'].append(1)
            # cv2.rectangle(crop_image, (int(bbox_x0),int(bbox_y0)), (int(bbox_x1), int(bbox_y1)), (0,0,255), 2)
    anno_info_list.append(gt_info)

with open(f'/workspace/humantracking/zhanhuireal2/ext_zhanhui_xianchang_2.json', 'w') as fp:
    json.dump(anno_info_list, fp)
print('sdf')

###################
# points = sample_anno_instance['value']['points']
# label_name = sample_anno_instance['value']['polygonlabels'][0]
# # label_id = label_name_and_label_id_map[label_name]

# points_array = np.array(points) 
# points_array[:, 0] = points_array[:, 0] / 100.0 * width
# points_array[:, 1] = points_array[:, 1] / 100.0 * height
# points_array = points_array.astype(np.int32)
# image = cv2.imread('/workspace/humantracking/bad-ball.png')
# mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
# mask = cv2.fillConvexPoly(mask, points_array, 255)
# cv2.imwrite('./bad-ball-v1.png', np.concatenate([image, np.expand_dims(mask, -1)], -1))
