from ochumanApi.ochuman import OCHuman
import cv2, os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15)
import ochumanApi.vis as vistool
from ochumanApi.ochuman import Poly2Mask
import numpy as np
import json

ochuman = OCHuman(AnnoFile='/workspace/dataset/OCHuman/ochuman.json', Filter='kpt&segm')
image_ids = ochuman.getImgIds()
print ('Total images: %d'%len(image_ids))

ImgDir = '/workspace/dataset/OCHuman/images'
target_folder = '/workspace/dataset/PoseSeg-OC'

anno_info = []
for image_id in image_ids:
    data = ochuman.loadImgs(imgIds=[image_id])[0]
    img = cv2.imread(os.path.join(ImgDir, data['file_name']))
    height, width = data['height'], data['width']

    image_pure_name = data['file_name'].split('.')[0]
    colors = [[255, 0, 0], 
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255], 
            [0, 0, 255], 
            [255, 0, 255]]

    for i, anno in enumerate(data['annotations']):
        bbox = anno['bbox']
        kpt = anno['keypoints']
        segm = anno['segms']
        max_iou = anno['max_iou']

        # img = vistool.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
        # if segm is not None:
        mask = Poly2Mask(segm)

        pos = np.where(mask > 0)
        min_y = np.min(pos[0])
        min_x = np.min(pos[1])
        max_y = np.max(pos[0])
        max_x = np.max(pos[1])

        min_y = max(min_y-20, 0)
        min_x = max(min_x-20, 0)
        max_y = min(max_y+20, height)
        max_x = min(max_x+20, width)

        kpt = np.array(kpt).reshape(-1, 3)
        kpt[:,0] = kpt[:,0] - min_x
        kpt[:,1] = kpt[:,1] - min_y
        kpt = kpt.tolist()
        # cv2.imwrite(f'{target_folder}/images/{image_pure_name}_{i}.png', img[min_y:max_y, min_x:max_x])
        # cv2.imwrite(f'{target_folder}/masks/{image_pure_name}_{i}.png', (mask[min_y:max_y, min_x:max_x]*255).astype(np.uint8))

        anno_info.append({
            'image': f'PoseSeg-OC/images/{image_pure_name}_{i}.png',
            'mask': f'PoseSeg-OC/masks/{image_pure_name}_{i}.png',
            'pose': kpt
        })

with open('./poseseg_oc.json', 'w') as fp:
    json.dump(anno_info, fp)

print('sdf')