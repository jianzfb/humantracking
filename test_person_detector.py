import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
import cv2
import numpy as np


def det_result_show(frame_index, image, obj_bboxes, obj_labels):
    for obj_bbox, obj_label in zip(obj_bboxes, obj_labels):
        x0,y0,x1,y1 = obj_bbox[:4]
        if int(obj_label) == 0:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        else:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)

    return image


op.load('detpostprocessop', '/workspace/project/sports/control')
video_dc['image', 'frame_index']('/workspace/1531_1721197037_raw.mp4'). \
    keep_ratio_op['image', ('keep_ratio_image_for_det', 'keep_ratio_bbox')](aspect_ratio=1.77). \
    resize_op['keep_ratio_image_for_det', 'resized_image_for_det'](out_size=(384,256)). \
    inference_onnx_op['resized_image_for_det', 'person_ball_feature_det'](
        mean=(128, 128, 128),
        std=(128, 128, 128),
        onnx_path='/workspace/humantracking/coco-mosaic_person_ball_epoch_160-model.onnx',
    ). \
    deploy.detpostprocess_func[('image', 'person_ball_feature_det'), ('obj_bboxes', 'obj_labels')](level_hw=np.array([32,48,16,24,8,12], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
    runas_op[('frame_index', 'image', 'obj_bboxes', 'obj_labels'), 'out'](func=det_result_show). \
    select('out').as_raw(). \
    to_video('./xyzabc.mp4')

