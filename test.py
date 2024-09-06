import os
import sys
import numpy as np
import cv2
sys.path.insert(0, '/workspace/antgo')

from antgo.framework.helper.dataset import *
tfd = TFDataset(
    ['/workspace/dataset/personfacehand/mainpfhv3'],
    pipeline=[
        dict(type='DecodeImage', to_rgb=False),
    ],
    description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
    inputs_def=dict(
        fields = ["image", 'bboxes', 'labels', 'image_meta']
    ),
    shuffle_queue_size=20
)


label_map = {
    0: (255,0,0),
    1: (0,255,0),
    2: (0,0,255)
}

count = 0
for data in tfd:
    image = data['image']
    bboxes = data['bboxes']
    labels = data['labels']

    if count > 1000:
        break

    # pos = np.where(labels == 1)
    # if len(pos[0]) > 1:
    #     print("erro")
    # if len(pos[0]) == 0:
    #     print('zero ball num')
    # else:
    #     print(f'ball num {len(pos[0])}')

    for bbox, label in zip(bboxes, labels):
        person_x0, person_y0, person_x1, person_y1 = bbox
        image = cv2.rectangle(image, (int(person_x0), int(person_y0)), (int(person_x1), int(person_y1)), label_map[label], 2)
    
    cv2.imwrite(f"/workspace/humantracking/CC/{count}.png", image)
    count += 1
