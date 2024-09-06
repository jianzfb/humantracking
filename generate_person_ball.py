import os
import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/workspace/antgo')
from antgo.dataflow.datasynth import *
from antgo.dataflow.dataset import *
from antgo.utils.sample_gt import *
ball_file_list = []

def generate_image():
    folder = '/workspace/dataset/ext/ball-ext-dataset/background'
    file_list = []
    for sub_folder_name in os.listdir(folder):
        if not sub_folder_name.startswith('spider'):
            continue
        folder_path = os.path.join(folder, sub_folder_name, 'test')
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            file_list.append(file_path)

    for i in range(20000):
        file_path_i = i % len(file_list)
        file_path = file_list[file_path_i]
        image = cv2.imread(file_path)

        image_h, image_w = image.shape[:2]

        # 蓝色垫子模板
        cushion_image_file = '/workspace/dataset/ext/ball-ext-dataset/imagewithmask/cushion_1.png'
        # 垫子
        cushion_image = cv2.imread(cushion_image_file, cv2.IMREAD_UNCHANGED)
        cushion_mask = cushion_image[:,:,3]
        cushion_image = cushion_image[:,:,:3]

        # 随机旋转
        angle = np.random.randint(0, 60 * 2) - 60
        cx = cushion_image.shape[1] / 2
        cy = cushion_image.shape[0] / 2
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cushion_image = cv2.warpAffine(
                src=cushion_image,
                M=rot_mat,
                dsize=cushion_image.shape[1::-1],
                flags=cv2.INTER_AREA)
        cushion_mask = cv2.warpAffine(
                cushion_mask,
                M=rot_mat,
                dsize=cushion_mask.shape[1::-1],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

        pos = np.where(cushion_mask > 128)
        y0 = np.min(pos[0])
        y1 = np.max(pos[0])
        x0 = np.min(pos[1])
        x1 = np.max(pos[1])

        cushion_image = cushion_image[y0:y1, x0:x1]
        cushion_mask = cushion_mask[y0:y1, x0:x1]

        cushion_h, cushion_w = cushion_image.shape[:2]
        scale = min(float(image_h)/float(cushion_h), float(image_w)/float(cushion_w))

        # 0.5~0.9
        random_scale = ((0.9-0.5)*np.random.random() + 0.5) * scale
        scaled_h = (int)(cushion_h * random_scale)
        scaled_w = (int)(cushion_w * random_scale)

        cushion_image = cv2.resize(cushion_image, (scaled_w, scaled_h))
        cushion_mask = cv2.resize(cushion_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        cushion_mask = cushion_mask/255
        cushion_mask = np.expand_dims(cushion_mask, -1)

        paste_x = int(np.random.randint(0, image_w - scaled_w))
        paste_y = int(np.random.randint(0, image_h - scaled_h))
        if np.random.random() < 0.7:
            image[paste_y:paste_y+scaled_h, paste_x:paste_x+scaled_w] = (1-cushion_mask) * image[paste_y:paste_y+scaled_h, paste_x:paste_x+scaled_w] + cushion_mask * cushion_image

        yield image


def generate_obj():
    # 球模板
    ball_template_file_list = []
    for filename in os.listdir('/workspace/dataset/ext/ball-ext-dataset/imagewithmask'):
        if filename[0] == '.':
            continue

        if 'cushion' in filename:
            continue
        ball_template_file_list.append(os.path.join('/workspace/dataset/ext/ball-ext-dataset/imagewithmask', filename))

    count = 0
    while True:
        template_num = len(ball_template_file_list)
        template_i = np.random.randint(0, template_num)

        template_file_path = ball_template_file_list[template_i]
        template_image = cv2.imread(template_file_path, cv2.IMREAD_UNCHANGED)
        h,w = template_image.shape[:2]
        mask = template_image[:,:,3]
        template_image = template_image[:,:,:3]

        pos = np.where(mask > 128)
        y0 = np.min(pos[0])
        y1 = np.max(pos[0])
        x0 = np.min(pos[1])
        x1 = np.max(pos[1])

        ball_image = template_image[y0:y1, x0:x1]
        ball_mask = mask[y0:y1, x0:x1]
        ball_h, ball_w = ball_image.shape[:2]

        # 球的宽度约为 0.5~0.9
        scale_value = 1.0
        ball_scaled_w = int(ball_w * scale_value)
        ball_scaled_h = int(ball_h * scale_value)
        ball_image = cv2.resize(ball_image, (ball_scaled_w, ball_scaled_h))
        ball_mask = cv2.resize(ball_mask, (ball_scaled_w, ball_scaled_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.concatenate([ball_image, np.expand_dims(ball_mask, -1)], axis=-1)
        yield {
            'image': canvas,
            'mask': ball_mask/255,
            'fill': 0
        }

        count += 1
        if count >= 20000:
            break


anno_info_list = []
count = 0
sgt = SampleGTTemplate()
for a,b in simple_syn_sample(generate_image(), generate_obj(), min_scale=0.06, max_scale=0.2):
    image = a
    obj_info = b

    cv2.imwrite(os.path.join('/workspace/dataset/ext/ball-ext-dataset/sync/', f'sync-{count}.png'), image)
    gt_info = sgt.get()    
    gt_info['image_file'] = os.path.join('/workspace/dataset/ext/ball-ext-dataset/sync/', f'sync-{count}.png')
    gt_info['height'] = image.shape[0]
    gt_info['width'] = image.shape[1]

    # pos = np.where(obj_info['mask'] == 1)
    # person_y0 = np.min(pos[0])
    # person_y1 = np.max(pos[0])
    # person_x0 = np.min(pos[1])
    # person_x1 = np.max(pos[1])

    pos = np.where(obj_info['mask'] == 1)
    ball_y0 = np.min(pos[0])
    ball_y1 = np.max(pos[0])
    ball_x0 = np.min(pos[1])
    ball_x1 = np.max(pos[1])

    gt_info['bboxes'] = [[float(ball_x0), float(ball_y0), float(ball_x1), float(ball_y1)]]
    gt_info['labels'] = [1]

    # cv2.rectangle(image, (int(ball_x0),int(ball_y0)), (int(ball_x1), int(ball_y1)), (0,255,0), 2)
    # cv2.imwrite('./a.png', image)
    count += 1
    anno_info_list.append(gt_info)

with open('/workspace/dataset/ext/ball-ext-dataset/sync_ext_zhanhui_ball.json', 'w') as fp:
    json.dump(anno_info_list, fp)
print('sdf')
