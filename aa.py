import os
import json
from antgo.utils.sample_gt import *
import imagesize

with open('/workspace/train_with_facebox_facekeypoint_zhuohua_handbox.json', 'r') as fp:
    content = json.load(fp)

sgt = SampleGTTemplate()
anno_info_list = []
dataset_name_list = []
for info in content:
    if info['dataset'] == 'zhuohua':
        continue

    if 'human_annotations' not in info or 'face_box' not in info:
        continue
    
    image_path = f'images/{info["dataset"]}/{info["image_id"]}'
    dataset_name_list.append(info["dataset"])

    print(image_path)
    gt_info = sgt.get()
    gt_info['image_file'] = image_path
    # gt_info['height'] = image.shape[0]
    # gt_info['width'] = image.shape[1]
    gt_info['bboxes'] = []
    gt_info['labels'] = []

    for person_id, person_bbox in info['human_annotations'].items():
        if person_bbox[2] - person_bbox[0] <= 1 or person_bbox[3] - person_bbox[1] <= 1:
            continue

        gt_info['bboxes'].append([float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])])
        gt_info['labels'].append(0)

    for face_id, face_bbox in info['face_box'].items():
        if face_bbox[2] - face_bbox[0] <= 1 or face_bbox[3] - face_bbox[1] <= 1:
            continue

        gt_info['bboxes'].append([float(face_bbox[0]), float(face_bbox[1]), float(face_bbox[2]), float(face_bbox[3])])
        gt_info['labels'].append(1)

    if 'left_hand' in info:
        for hand_id, hand_bbox in info['left_hand'].items():
            if hand_bbox[2] - hand_bbox[0] <= 1 or hand_bbox[3] - hand_bbox[1] <= 1:
                continue

            gt_info['bboxes'].append([float(hand_bbox[0]), float(hand_bbox[1]), float(hand_bbox[2]), float(hand_bbox[3])])
            gt_info['labels'].append(2)

    if 'right_hand' in info:
        for hand_id, hand_bbox in info['right_hand'].items():
            if hand_bbox[2] - hand_bbox[0] <= 1 or hand_bbox[3] - hand_bbox[1] <= 1:
                continue

            gt_info['bboxes'].append([float(hand_bbox[0]), float(hand_bbox[1]), float(hand_bbox[2]), float(hand_bbox[3])])
            gt_info['labels'].append(3)

    anno_info_list.append(gt_info)


dataset_name_set = set(dataset_name_list)
print(dataset_name_set)
with open('./person_face_hand.json', 'w') as fp:
    json.dump(anno_info_list, fp)



#############################################
# with open('/workspace/hand_person_face_3w.json', 'r') as fp:
#     content = json.load(fp)

# sgt = SampleGTTemplate()
# anno_info_list = []
# for info in content['root']:    
#     image_path = f'image_all/{info["img_path"]}'

#     print(image_path)
#     gt_info = sgt.get()
#     gt_info['image_file'] = image_path
#     # gt_info['height'] = image.shape[0]
#     # gt_info['width'] = image.shape[1]
#     gt_info['bboxes'] = []
#     gt_info['labels'] = []

#     for person_bbox in info['person_bbox']:
#         x,y,w,h = person_bbox
#         gt_info['bboxes'].append([float(x), float(y), float(x+w), float(y+h)])
#         gt_info['labels'].append(0)

#     for face_bbox in info['face_box']:
#         x,y,w,h = face_bbox
#         gt_info['bboxes'].append([float(x), float(y), float(x+w), float(y+h)])
#         gt_info['labels'].append(1)

#     for hand_bbox in info['hand_bbox']:
#         x,y,w,h = hand_bbox
#         gt_info['bboxes'].append([float(x), float(y), float(x+w), float(y+h)])
#         gt_info['labels'].append(2)
#     anno_info_list.append(gt_info)

# with open('./person_face_hand_v2.json', 'w') as fp:
#     json.dump(anno_info_list, fp)


#####################
# part 3
# folder = '/workspace/xiaodu_anno'
# sgt = SampleGTTemplate()
# anno_info_list = []
# ignore_num = 0
# for anno_file in os.listdir(folder):
#     if anno_file[0] == '.':
#         continue

#     subfolder = anno_file.split('.')[0]
#     anno_path = os.path.join(folder, anno_file)
#     with open(anno_path, 'r') as fp:
#         content = fp.readline()
#         content = content.strip()
#         while content:
#             info = json.loads(content)

#             image_path = f'images/X3_xiaodu_trainset_part1/{subfolder}/data/{info["image_key"]}'
#             print(image_path)
#             gt_info = sgt.get()
#             gt_info['image_file'] = image_path
#             # gt_info['height'] = image.shape[0]
#             # gt_info['width'] = image.shape[1]
#             gt_info['bboxes'] = []
#             gt_info['labels'] = []

#             if 'person' in info:
#                 for person_info in info['person']:
#                     person_bbox = person_info['data']
#                     if person_bbox[2] - person_bbox[0] <= 1 or person_bbox[3] - person_bbox[1] <= 1:
#                         continue

#                     gt_info['bboxes'].append([float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])])
#                     gt_info['labels'].append(0)

#             if 'face' in info:
#                 for person_info in info['face']:
#                     person_bbox = person_info['data']
#                     if person_bbox[2] - person_bbox[0] <= 1 or person_bbox[3] - person_bbox[1] <= 1:
#                         continue

#                     gt_info['bboxes'].append([float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])])
#                     gt_info['labels'].append(1)

#             is_ignore = False
#             if 'hand' in info:
#                 for person_info in info['hand']:
#                     person_bbox = person_info['data']
#                     if person_bbox[2] - person_bbox[0] <= 1 or person_bbox[3] - person_bbox[1] <= 1:
#                         continue

#                     gt_info['bboxes'].append([float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])])
#                     if person_info['attrs']['lefthand_or_righthand'] == 'lefthand':
#                         gt_info['labels'].append(2)
#                     elif person_info['attrs']['lefthand_or_righthand'] == 'righthand':
#                         gt_info['labels'].append(3)
#                     else:
#                         print("why")
#                         is_ignore = True
#                         ignore_num += 1

#             content = fp.readline()
#             content = content.strip()

#             if is_ignore:
#                 continue

#             anno_info_list.append(gt_info)

        
# print(len(anno_info_list))
# print(f'ignore num {ignore_num}')
# with open('./person_face_hand_v3.json', 'w') as fp:
#     json.dump(anno_info_list, fp)