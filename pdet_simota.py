import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
from antgo.pipeline.extent import op
import cv2
import numpy as np


info = {}
def debug_show(prefix, image, bboxes, labels):
    global info
    image_h, image_w = image.shape[:2]
    # prefix = os.path.basename(prefix).replace('.png','')
    info[prefix] = {'file': [], 'bbox': []}
    for bbox, label in zip(bboxes, labels):
        if label == 0:
            x0,y0,x1,y1,_ = bbox.astype(np.int32)
            x0 = max(x0 - 1, 0)
            y0 = max(y0 - 1, 0)
            x1 = min(x1 + 1, image_w)
            y1 = min(y1 + 1, image_h)

            cv2.rectangle(image, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 2)
        if label == 1:
            x0,y0,x1,y1,_ = bbox.astype(np.int32)
            x0 = max(x0 - 1, 0)
            y0 = max(y0 - 1, 0)
            x1 = min(x1 + 1, image_w)
            y1 = min(y1 + 1, image_h)

            cv2.rectangle(image, (int(x0),int(y0)), (int(x1),int(y1)), (0,0,255), 2)
    
    # cv2.imwrite(os.path.join('/workspace/dataset/pole/test/', f'frame_{prefix}.png'), image)
    cv2.imwrite(os.path.join('./AA', f'frame_{prefix}.png'), image)
    print(f'{prefix}')


# video_dc['image']('/workspace/project/sports/volleyball/9_20230531_1_24.mp4'). \
#     select('out').as_raw().to_video(output_path='./aa.mp4', width=1920, height=1080)
# glob['file_path']('/workspace/dataset/test/*.png'). \
#     image_decode['file_path', 'image'](). \
op.load('detpostprocessop', '/workspace/project/leshi')

# 测试杆子检测
op.load('poledetpostprocessop', '/workspace/project/sports/basketball')
video_dc['image', 'frame_index']('/workspace/project/sports/video/basketball/20231226114906.ts'). \
    resize_op['image', 'resized_image'](out_size=(704,384)). \
    inference_onnx_op['resized_image', 'output'](
        onnx_path='/workspace/humantracking/coco-pole-epoch_80-model.onnx', 
        mean=[128, 128, 128],
        std=[128, 128, 128]
    ). \
    deploy.PoleDetPostProcessOp[('image', 'output'), ('bboxes', 'labels')](level_hw=np.array([48,88,24,44,12,22], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
    runas_op[("frame_index", "image", "bboxes", 'labels'), 'out'](func=debug_show).run()


# # 测试人+人脸检测
# video_dc['image', 'frame_index']('/workspace/project/sports/video/volleyball/246_1700214450.mp4'). \
#     resize_op['image', 'resized_image'](out_size=(256,192)). \
#     inference_onnx_op['resized_image', 'output'](
#         onnx_path='/workspace/humantracking/coco-epoch_80-model.onnx', 
#         mean=[128, 128, 128],
#         std=[128, 128, 128]
#     ). \
#     deploy.detpostprocess_func[('image', 'output'), ('bboxes', 'labels')](level_hw=np.array([24,32,12,16,6,8], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[("frame_index", "image", "bboxes", 'labels'), 'out'](func=debug_show).run()
