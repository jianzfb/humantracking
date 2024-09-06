import os
import sys
import cv2
from antgo.utils.sample_gt import *
from antgo.dataflow.dataset.coco2017 import CocoAPI

data_name = 'Basketball.v2i.coco'
data_flag = 'valid'
cls_name = ['ball']
coco_api = CocoAPI(f'/workspace/dataset/person-and-ball-test/{data_name}/{data_flag}/_annotations.coco.json')

cat_ids = coco_api.getCatIds(catNms=cls_name)
img_ids = coco_api.getImgIds(catIds=cat_ids)
# cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
print(len(img_ids))
anno_list = []
sgt = SampleGTTemplate()
for id in range(len(img_ids)):
    img_obj = coco_api.loadImgs(img_ids[id])[0]
    image_file = img_obj['file_name']
    annotation_ids = coco_api.getAnnIds(imgIds=img_obj['id'])
    annotation = coco_api.loadAnns(annotation_ids)

    boxes = []
    category_id = []

    image_path = os.path.join(f'/workspace/dataset/person-and-ball-test/{data_name}/{data_flag}', image_file)
    image = cv2.imread(image_path)    
    for ix, obj in enumerate(annotation):
        x, y, w, h = obj['bbox']

        if obj['category_id'] != 1:
            continue

        # 目标框
        boxes.append([x, y, x + w, y + h])
        # 目标类别
        category_id.append(1)
        print(f"category_id: {obj['category_id']}")

        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0),2)

    cv2.imwrite("./a.png", image)
    gt_info = sgt.get()    
    gt_info['image_file'] = image_path
    gt_info['height'] = image.shape[0]
    gt_info['width'] = image.shape[1]
    gt_info['bboxes'] = boxes
    gt_info['labels'] = category_id

    anno_list.append(gt_info)
    
with open(f'/workspace/dataset/person-and-ball-test/{data_name}_{data_flag}.json', 'w') as fp:
    json.dump(anno_list, fp)