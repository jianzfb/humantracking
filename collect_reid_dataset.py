import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
import cv2
import numpy as np
from antgo.pipeline.extent import op
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *


def select_face_boxes(image, obj_bboxes, obj_labels):
    face_label_mask = obj_labels == 1
    face_bboxes = obj_bboxes[face_label_mask]
    if face_bboxes.shape[0] > 0:
        return face_bboxes[0].astype(np.float32)

    return np.empty((0, 5), dtype=np.float32)


def select_person_boxes(image, obj_bboxes, obj_labels):
    face_label_mask = obj_labels == 0
    face_bboxes = obj_bboxes[face_label_mask]
    if face_bboxes.shape[0] > 0:
        x0,y0,x1,y1 = face_bboxes[0][:4]
        image_h, image_w = image.shape[:2]
        x0 = max(int(x0), 0)
        y0 = max(int(y0), 0)
        x1 = min(int(x1), image_w)
        y1 = min(int(y1), image_h)

        person_image = image[y0:y1, x0:x1]
        rhi, rwi = person_image.shape[:2]
        aspect_ratio = 0.5
        nhi, nwi = rhi, rwi
        if abs(rwi / rhi - aspect_ratio) > 0.0001:
            if rwi / rhi > aspect_ratio:
                nwi = rwi
                nhi = int(rwi / aspect_ratio)
            else:
                nhi = rhi
                nwi = int(rhi * aspect_ratio)

        new_image = np.zeros((nhi, nwi,3), dtype=np.uint8)

        offset_x = (nwi - person_image.shape[1]) // 2
        new_image[:rhi, offset_x:rwi+offset_x] = person_image
        person_image = cv2.resize(new_image, (128,256))        
        person_image = person_image[:,:,::-1]
        return face_bboxes[0].astype(np.float32), person_image

    return np.empty((0, 5), dtype=np.float32), np.empty((0,0,3), dtype=np.uint8)


def test_face(face_bbox, face_image):
    if face_bbox.shape[0] == 0:
        return

    # for face_i in range(face_image.shape[0]):
    #     one_face_image = face_image[face_i]
    #     cv2.imwrite(f'./face_{face_i}.png')
    cv2.imwrite(f'./face.png', face_image)
    print(face_image.shape)


face_factory = {}
def analyze_face_and_collect(image, obj_bboxes, obj_labels, face_bboxes, face_image, face_feature):
    if face_bboxes.shape[0] == 0:
        return

    face_x0, face_y0, face_x1, face_y1 = face_bboxes[:4]
    face_cx = (face_x0+face_x1)/2.0
    face_cy = (face_y0+face_y1)/2.0

    image_h, image_w = image.shape[:2]
    # 找到与此人对应的人体
    global face_factory
    for obj_i, (obj_bbox, obj_label) in enumerate(zip(obj_bboxes, obj_labels)):
        if obj_label != 0:
            continue

        person_x0, person_y0, person_x1, person_y1 = obj_bbox[:4]
        face_id = -1
        if face_x0 > person_x0 and face_x1 < person_x1 and face_y0 > person_y0:
            is_found = False
            for face_id_in_record, face_feature_in_record in face_factory.items():
                face_sim = face_feature @ face_feature_in_record.transpose()
                if face_sim[0,0] > 0.5:
                    is_found = True
                    face_id = face_id_in_record
                    break

            if not is_found:
                face_id = len(face_factory)
                face_factory[face_id] = face_feature

            root_folder = '/workspace/dataset/beta-reid-dataset/train'
            face_id_folder = f'{root_folder}/{face_id}/'
            if not os.path.exists(face_id_folder):
                os.makedirs(face_id_folder)
            file_num = len(os.listdir(face_id_folder))

            ext_person_x0 = max(person_x0-5, 0)
            ext_person_y0 = max(person_y0-5, 0)

            ext_person_x1 = min(person_x1+5, image_w)
            ext_person_y1 = min(person_y1+5, image_h)
            person_image = image[int(ext_person_y0):int(ext_person_y1), int(ext_person_x0):int(ext_person_x1)].copy()
            cv2.imwrite(os.path.join(face_id_folder, f'{file_num+1}.png'), person_image)
            break


person_factory = {}
def analyze_person_and_collect(image, obj_bboxes, obj_labels, person_bboxes, person_images, person_features):
    if person_bboxes.shape[0] == 0:
        return

    person_x0, person_y0, person_x1, person_y1 = person_bboxes[:4]
    person_cx = (person_x0+person_x1)/2.0
    person_cy = (person_y0+person_y1)/2.0

    if (person_y1-person_y0)/(person_x1-person_x0) < 2.2:
        # 忽略非全身
        return

    image_h, image_w = image.shape[:2]
    
    global person_factory
    is_found = False
    person_id = -1

    best_person_id = -1
    best_match_score = 0.0
    for person_id_in_record, person_feature_in_record in person_factory.items():
        person_sim = person_features @ person_feature_in_record.transpose()
        if person_sim[0,0] > 0.4 and person_sim[0,0] > best_match_score:
            is_found = True
            person_id = person_id_in_record
            best_match_score = person_sim[0,0]

    if not is_found:
        person_id = len(person_factory)
        person_factory[person_id] = person_features

    root_folder = '/workspace/dataset/beta-reid-dataset/train3'
    person_id_folder = f'{root_folder}/{person_id}/'
    if not os.path.exists(person_id_folder):
        os.makedirs(person_id_folder)

    ext_person_x0 = max(person_x0-5, 0)
    ext_person_y0 = max(person_y0-5, 0)

    ext_person_x1 = min(person_x1+5, image_w)
    ext_person_y1 = min(person_y1+5, image_h)
    person_image = image[int(ext_person_y0):int(ext_person_y1), int(ext_person_x0):int(ext_person_x1)].copy()

    file_num = len(os.listdir(person_id_folder))
    cv2.imwrite(os.path.join(person_id_folder, f'{file_num+1}.png'), person_image)


op.load('detpostprocessop', '/workspace/project/sports/control')

# # 使用人脸模型
# camera_dc['image', 'frame_index']('rtsp://admin:a123.123456@192.168.1.223:554/cam/stream0'). \
#     resize_op['image', 'resized_image_for_person'](out_size=(384,256)). \
#     inference_onnx_op['resized_image_for_person', 'peson_output'](
#         onnx_path='/workspace/humantracking/coco-person_face_256_384-model.onnx', 
#         mean=[128, 128, 128],
#         std=[128, 128, 128]
#     ). \
#     deploy.detpostprocess_func[('image', 'peson_output'), ('obj_bboxes', 'obj_labels')](level_hw=np.array([32,48,16,24,8,12], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[('image', 'obj_bboxes', 'obj_labels'), 'face_bboxes'](func=select_face_boxes). \
#     eagleeye.op.FaceAlignOp[('image', 'face_bboxes'), 'face_image'](target_h=160, target_w=160, margin=0). \
#     runas_op[('face_bboxes', 'face_image'), 'yy'](func=test_face). \
#     inference_onnx_op['face_image', 'face_feature'](
#         onnx_path='/workspace/project/facenetv/facenet-epoch_92-model.onnx', 
#         mean=[127.5, 127.5, 127.5],
#         std=[128, 128, 128],
#     ). \
#     runas_op[('image', 'obj_bboxes', 'obj_labels', 'face_bboxes', 'face_image', 'face_feature'), 'out'](func=analyze_face_and_collect).run()

# 使用REID模型
camera_dc['image', 'frame_index']('rtsp://admin:a123.123456@192.168.1.223:554/cam/stream0'). \
    resize_op['image', 'resized_image_for_person'](out_size=(384,256)). \
    inference_onnx_op['resized_image_for_person', 'peson_output'](
        onnx_path='/workspace/humantracking/coco-epoch_80_new-model.onnx', 
        mean=[128, 128, 128],
        std=[128, 128, 128]
    ). \
    deploy.detpostprocess_func[('image', 'peson_output'), ('obj_bboxes', 'obj_labels')](level_hw=np.array([32,48,16,24,8,12], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
    runas_op[('image', 'obj_bboxes', 'obj_labels'), ('person_bboxes', 'person_images')](func=select_person_boxes). \
    inference_onnx_op['person_images', 'person_features'](
        onnx_path='/workspace/humantracking/reid-epoch_75-model.onnx', 
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
    ). \
    runas_op[('image', 'obj_bboxes', 'obj_labels', 'person_bboxes', 'person_images', 'person_features'), 'out'](func=analyze_person_and_collect).run()


# # 测试rtsp
# camera_dc['image', 'frame_index']('rtsp://admin:a123.123456@192.168.1.223:554/cam/stream0'). \
#     rtsp(frame_key='image', rtsp_ip='192.168.1.90')

# print('sf')