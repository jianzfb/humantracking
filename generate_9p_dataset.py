import os
import cv2
import numpy as np
import json

ahp_folder = '/workspace/dataset/person-seg/PoseSeg-AHP'
ahp_anno_file = os.path.join(ahp_folder, 'poseseg_ahp_train_new.json')
ahp_anno_info = []
with open(ahp_anno_file, 'r') as fp:
    content = json.load(fp)
    ahp_anno_info.extend(content)

# for a in ahp_anno_info:
#     a['image'] = '/'.join(a['image'].split('/')[1:])
#     a['mask'] = '/'.join(a['mask'].split('/')[1:])

# with open(os.path.join(ahp_folder, 'poseseg_ahp_val_new.json'), 'w') as fp:
#     json.dump(ahp_anno_info, fp)

ahp_num = len(ahp_anno_info)

sudoku_anno_info = []
generate_sample_num = 10000
for sample_index in range(generate_sample_num):
    print(sample_index)
    person_num = np.random.randint(1,9)

    person_image_list = []
    person_mask_list = []
    for person_i in range(person_num):
        id = np.random.randint(ahp_num)

        anno_info = ahp_anno_info[id]
        image = cv2.imread(os.path.join('/workspace/dataset/person-seg/PoseSeg-AHP', anno_info['image']))
        image_h, image_w = image.shape[:2]
        mask = cv2.imread(os.path.join('/workspace/dataset/person-seg/PoseSeg-AHP', anno_info['mask']), cv2.IMREAD_GRAYSCALE)
        mask = mask/255
        mask = mask.astype(np.uint8)

        person_pos = np.where(mask == 1)
        x0 = np.min(person_pos[1])
        y0 = np.min(person_pos[0])

        x1 = np.max(person_pos[1])
        y1 = np.max(person_pos[0])

        random_ext_size = int(np.random.randint(5, 20))
        x0 = int(max(0, x0-random_ext_size))
        y0 = int(max(0, y0-random_ext_size))
        x1 = int(min(x1+random_ext_size, image_w))
        y1 = int(min(y1+random_ext_size, image_h))

        person_image = image[y0:y1,x0:x1].copy()
        person_h,person_w = person_image.shape[:2]
        person_mask = mask[y0:y1,x0:x1].copy()

        person_image_list.append(person_image)
        person_mask_list.append(person_mask)
    
    image_canvas = np.zeros((480, 384, 3), dtype=np.uint8)
    mask_canvas = np.zeros((480, 384), dtype=np.uint8)

    if person_num == 1:
        image_canvas = cv2.resize(person_image_list[0], (384, 480))
        mask_canvas = cv2.resize(person_mask_list[0], (384, 480))
    elif person_num == 2:
        resized_person_image = cv2.resize(person_image_list[0], (384//2, 480))
        resized_person_mask = cv2.resize(person_mask_list[0], (384//2, 480))

        image_canvas[:480, :384//2] = resized_person_image
        mask_canvas[:480, :384//2] = resized_person_mask

        resized_person_image = cv2.resize(person_image_list[1], (384//2, 480))
        resized_person_mask = cv2.resize(person_mask_list[1], (384//2, 480))

        image_canvas[:480, 384//2:] = resized_person_image
        mask_canvas[:480, 384//2:] = resized_person_mask
    elif person_num == 3:
        resized_person_image = cv2.resize(person_image_list[0], (384//3, 480))
        resized_person_mask = cv2.resize(person_mask_list[0], (384//3, 480))

        image_canvas[:480, :384//3] = resized_person_image
        mask_canvas[:480, :384//3] = resized_person_mask

        resized_person_image = cv2.resize(person_image_list[1], (384//3, 480))
        resized_person_mask = cv2.resize(person_mask_list[1], (384//3, 480))

        image_canvas[:480, 384//3:384//3*2] = resized_person_image
        mask_canvas[:480, 384//3:384//3*2] = resized_person_mask

        resized_person_image = cv2.resize(person_image_list[2], (384//3, 480))
        resized_person_mask = cv2.resize(person_mask_list[2], (384//3, 480))

        image_canvas[:480, 384//3*2:] = resized_person_image
        mask_canvas[:480, 384//3*2:] = resized_person_mask
    elif person_num == 4:
        for person_i, (person_image, person_mask) in enumerate(zip(person_image_list, person_mask_list)):
            row_i = person_i // 2
            col_i = person_i - person_i // 2 * 2

            resized_person_image = cv2.resize(person_image, (192, 240))
            resized_person_mask = cv2.resize(person_mask, (192, 240))

            image_canvas[row_i*240:(row_i+1)*240, col_i*192:(col_i+1)*192] = resized_person_image
            mask_canvas[row_i*240:(row_i+1)*240, col_i*192:(col_i+1)*192] = resized_person_mask
    else:
        for person_i, (person_image, person_mask) in enumerate(zip(person_image_list, person_mask_list)):
            row_i = person_i // 3
            col_i = person_i - person_i // 3 * 3

            resized_person_image = cv2.resize(person_image, (128, 160))
            resized_person_mask = cv2.resize(person_mask, (128, 160))

            image_canvas[row_i*160:(row_i+1)*160, col_i*128:(col_i+1)*128] = resized_person_image
            mask_canvas[row_i*160:(row_i+1)*160, col_i*128:(col_i+1)*128] = resized_person_mask

    cv2.imwrite(f'/workspace/dataset/person-seg/sudoku/images/{sample_index}.png', image_canvas)
    cv2.imwrite(f'/workspace/dataset/person-seg/sudoku/masks/{sample_index}.png', mask_canvas)

    sudoku_anno_info.append({
        'image': f'images/{sample_index}.png',
        'mask': f'masks/{sample_index}.png'
    })


# train_num = int(len(sudoku_anno_info)*0.9)
# test_num = len(sudoku_anno_info) - train_num
# train_sudoku_anno_info = sudoku_anno_info[:train_num]
# test_sudoku_anno_info = sudoku_anno_info[train_num:]
# with open('/workspace/dataset/person-seg/sudoku/train.json', 'w') as fp:
#     json.dump(train_sudoku_anno_info, fp)

# with open('/workspace/dataset/person-seg/sudoku/test.json', 'w') as fp:
#     json.dump(test_sudoku_anno_info, fp)
