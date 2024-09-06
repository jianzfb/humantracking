import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
import cv2
import numpy as np


def det_result_show(frame_index, person_ball_feature_det,image, obj_bboxes, obj_labels):
    for obj_bbox, obj_label in zip(obj_bboxes, obj_labels):
        x0,y0,x1,y1 = obj_bbox[:4]
        if int(obj_label) == 0:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        else:
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)

    # cv2.imwrite(f'./temp/{frame_index}.png', image)
    # print('sdf')
    cv2.imwrite('./aa111.png', image)
    return image


op.load('detpostprocessop', '/workspace/humantracking/extent')
# video_dc['image', 'frame_index']('/workspace/humantracking/2024_07_04_18_42_45.mp4'). \
# image = cv2.imread('/workspace/humantracking/VCG41200223805-001.jpg')
# placeholder['image', 'frame_index'](image, 0). \

# 测试512,768模型
# video_dc['image', 'frame_index']('/workspace/humantracking/jumprope_jump.mov'). \
#     resize_op['image', 'resized_image_for_det'](out_size=(768,512)). \
#     inference_onnx_op['resized_image_for_det', 'person_ball_feature_det'](
#         mean=(128, 128, 128),
#         std=(128, 128, 128),
#         onnx_path='/workspace/humantracking/person_ball_best_512_768.onnx',
#     ). \
#     deploy.detpostprocess_func[('image', 'person_ball_feature_det'), ('obj_bboxes', 'obj_labels')](model_size=np.array([512,768], dtype=np.int32),level_hw=np.array([64,96,32,48,16,24], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[('frame_index', 'person_ball_feature_det', 'image', 'obj_bboxes', 'obj_labels'), 'out'](func=det_result_show). \
#     select('out').as_raw(). \
#     to_video('./hh.mp4')


# # 测试256，384模型
# # video_dc['image', 'frame_index']('/workspace/humantracking/1531_1721197037.mp4'). \
# imread_dc['image']('/workspace/humantracking/1661723801299_.pic_hd.jpg'). \
#     placeholder_int32_op['frame_index'](0). \
#     resize_op['image', 'resized_image_for_det'](out_size=(384,256)). \
#     inference_onnx_op['resized_image_for_det', 'person_ball_feature_det'](
#         mean=(128, 128, 128),
#         std=(128, 128, 128),
#         onnx_path='/workspace/project/sports/sdk/deploy/sports_plugin/model/personball_best_1_0.onnx',
#     ). \
#     deploy.detpostprocess_func[('image', 'person_ball_feature_det'), ('obj_bboxes', 'obj_labels')](model_size=np.array([256,384], dtype=np.int32), level_hw=np.array([32,48,16,24,8,12], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[('frame_index', 'person_ball_feature_det', 'image', 'obj_bboxes', 'obj_labels'), 'out'](func=det_result_show).run()
#     # select('out').as_raw(). \
#     # to_video('./xyzabc.mp4')


# 测试256，384模型
# video_dc['image', 'frame_index']('/workspace/humantracking/1531_1721197037.mp4'). \


imread_dc['image']('/workspace/humantracking/18511724918617_.pic.jpg'). \
    placeholder_int32_op['frame_index'](0). \
    resize_op['image', 'resized_image_for_det'](out_size=(768,512)). \
    inference_onnx_op['resized_image_for_det', 'person_ball_feature_det'](
        mean=(128, 128, 128),
        std=(128, 128, 128),
        onnx_path='/workspace/humantracking/person_ball_best_512_768.onnx',
    ). \
    deploy.detpostprocess_func[('image', 'person_ball_feature_det'), ('obj_bboxes', 'obj_labels')](model_size=np.array([512,768], dtype=np.int32), level_hw=np.array([64,96,32,48,16,24], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
    runas_op[('frame_index', 'person_ball_feature_det', 'image', 'obj_bboxes', 'obj_labels'), 'out'](func=det_result_show).run()
    # select('out').as_raw(). \
    # to_video('./xyzabc.mp4')
