import os
import json
from antgo.utils.sample_gt import *
import cv2
anno_file = '/workspace/dataset/merge/annot/train_coco_mpii_aic_lsp_mhp_aichallenge.json'
with open(anno_file, 'r') as fp:
    info = json.load(fp)
    
sgt = SampleGTTemplate()

root = '/workspace/dataset/merge/images'
anno_list = []
for sample_info in info:
    image_id = sample_info['image_id']
    image_path = os.path.join(root, sample_info['dataset'], image_id)

    image = cv2.imread(image_path)
    obj_boxes = []
    obj_label = []
    for k,v in sample_info['human_annotations'].items():
        obj_boxes.append(v)
        obj_label.append(0)
    
    if 'face_box' not in sample_info:
        print(f'No face box {image_path}')
        continue

    print(image_path)
    for k,v in sample_info['face_box'].items():
        if len(v) < 4:
            continue
        if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 and v[3] == 0.0:
            continue        
        if v[2] - v[0] < 1 or v[3] - v[1] < 1:
            continue

        obj_boxes.append(v)
        obj_label.append(1)

    # label_map = {
    #     0: (255,0,0),
    #     1: (0,255,0)
    # }
    # for bbox, label in zip(obj_boxes, obj_label):
    #     x0,y0,x1,y1 = bbox
    #     image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), label_map[label], 2)
    # cv2.imwrite('./a.png', image)

    gt_info = sgt.get()    
    gt_info['image_file'] = image_path
    gt_info['height'] = image.shape[0]
    gt_info['width'] = image.shape[1]
    gt_info['bboxes'] = obj_boxes
    gt_info['labels'] = obj_label

    anno_list.append(gt_info)


with open('./large_person_and_face.json', 'w') as fp:
    json.dump(anno_list, fp)