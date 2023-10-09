from ochumanApi.ochuman import OCHuman
import cv2, os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15)
import ochumanApi.vis as vistool
from ochumanApi.ochuman import Poly2Mask
import numpy as np
import json

# ochuman = OCHuman(AnnoFile='/workspace/dataset/OCHuman/ochuman.json', Filter='kpt&segm')
# image_ids = ochuman.getImgIds()
# print ('Total images: %d'%len(image_ids))

folder = '/workspace/dataset/AHP'
image_folder = os.path.join(folder, 'train', 'JPEGImages')
ann_folder = os.path.join(folder, 'train', 'Annotations')

target_folder = '/workspace/dataset/PoseSeg-AHP'

anno_info = []
for line in open(os.path.join(folder, 'train','ImageSets', 'val.txt')):
    image_pure_name = line.strip()
    image_file = os.path.join(image_folder, f'{image_pure_name}.jpg')
    mask_file = os.path.join(ann_folder, f'{image_pure_name}.png')
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file)[:,:,2]

    height,width = image.shape[:2]
    pos = np.where(mask > 0)
    min_y = np.min(pos[0])
    min_x = np.min(pos[1])
    max_y = np.max(pos[0])
    max_x = np.max(pos[1])

    min_y = max(min_y-25, 0)
    min_x = max(min_x-25, 0)
    max_y = min(max_y+25, height)
    max_x = min(max_x+25, width)
    mask[pos] = 1

    cv2.imwrite(f'{target_folder}/images/{image_pure_name}.png', image[min_y:max_y, min_x:max_x])
    cv2.imwrite(f'{target_folder}/masks/{image_pure_name}.png', (mask[min_y:max_y, min_x:max_x]*255).astype(np.uint8))

    anno_info.append({
        'image': f'PoseSeg-AHP/images/{image_pure_name}.png',
        'mask': f'PoseSeg-AHP/masks/{image_pure_name}.png',
    })

with open('./poseseg_ahp_val.json', 'w') as fp:
    json.dump(anno_info, fp)

print('sdf')