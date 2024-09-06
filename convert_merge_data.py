import os
import json
import cv2
import numpy as np

with open('/workspace/dataset/train_with_facebox_facekeypoint_zhuohua_handbox_update4mediapipeOrder_230818.json', 'r') as fp:
    content = json.load(fp)

    for i in range(100):
        image_id = content[i]['image_id']
        image_dataset = content[i]['dataset']

        folder = '/workspace/dataset'
        image_path = os.path.join(folder, image_dataset, image_id)
        image = cv2.imread(image_path)

        person_num = len(content[i]['keypoint_annotations'])
        random_person_i = np.random.choice(person_num)
        keys = list(content[i]['keypoint_annotations'].keys())
        random_person_key = keys[random_person_i]
        random_person_points = content[i]['keypoint_annotations'][random_person_key]
        random_person_points = np.array(random_person_points)
        x0 = np.min(random_person_points[:,0])
        y0 = np.min(random_person_points[:,1])
        x1 = np.max(random_person_points[:,0])
        y1 = np.max(random_person_points[:,1])

        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2, 2)    
        joint_i = 0 
        for x,y, score in random_person_points:
            cv2.circle(image, (int(x), int(y)), 2, (0,0,255), 2)
            cv2.putText(image, f'{joint_i}', (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            joint_i += 1

        cv2.imwrite('./a.png', image)
        image_id = content[i]
        print('ss')