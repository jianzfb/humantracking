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
    # if 'coco' in sample_info['dataset'].lower():
    #     continue
    # if 'aic' in sample_info['dataset'].lower():
    #     continue
    # if 'keypoint_train_images_20170902' in sample_info['dataset'].lower():
    #     continue
    
    image_id = sample_info['image_id']
    image_path = os.path.join(root, sample_info['dataset'], image_id)
    print(image_path)
    image = cv2.imread(image_path)
    obj_boxes = []
    obj_label = []
    for k,v in sample_info['human_annotations'].items():
        obj_boxes.append(v)
        obj_label.append(0)
    
    # 查找当前样本是否存在 球的标注信息
    file_id = f'{sample_info["dataset"]}/{image_id}'
    file_id = file_id.replace('/', '-')
    ext_name = file_id.split('.')[-1]
    file_id = file_id.replace(f'.{ext_name}', '.txt')    

    if os.path.exists(os.path.join('/workspace/dataset/merge/temp', file_id)):
        for line in open(os.path.join('/workspace/dataset/merge/temp', file_id)):
            x0,y0,x1,y1 = line.split(' ')
            obj_boxes.append([float(x0), float(y0), float(x1), float(y1)])
            obj_label.append(1)

    # for bbox in obj_boxes:
    #     x0,y0,x1,y1 = bbox
    #     image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
    # cv2.imwrite('./a.png', image)

    gt_info = sgt.get()    
    gt_info['image_file'] = image_path
    gt_info['height'] = image.shape[0]
    gt_info['width'] = image.shape[1]
    gt_info['bboxes'] = obj_boxes
    gt_info['labels'] = obj_label
    
    anno_list.append(gt_info)

with open('./large_person_and_ball.json', 'w') as fp:
    json.dump(anno_list, fp)