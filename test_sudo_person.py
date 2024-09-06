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


def debug_show(person_image, parsing):
    result = parsing[0,0]
    # parsing_idx = np.argmax(result, 0)
    
    parsing_mask = (result*255).astype(np.uint8)
    h,w = person_image.shape[:2]
    parsing_mask = cv2.resize(parsing_mask, (w,h))

    canvas = np.zeros((h,w*2,3), dtype=np.uint8)
    canvas[:,:w] = person_image
    canvas[:,w:,0] = parsing_mask
    canvas[:,w:,1] = parsing_mask
    canvas[:,w:,2] = parsing_mask

    cv2.imwrite('./a.png',canvas)
    return None


def select_person(image, obj_bboxes, obj_labels):
    index = obj_labels==0
    person_bboxes = obj_bboxes[index]
    x0,y0,x1,y1,_ = person_bboxes[0].astype(np.int32)
    person_image = image[y0-10:y1+30,x0-20:x1+20].copy()

    return person_image

# personseg-epoch_120-model.onnx

op.load('detpostprocessop', '/workspace/project/sports/control')
# video_dc['image', 'frame_index']('/workspace/project/sports/video/longjump/242_1700041812.mp4'). \
#     resize_op['image', 'resized_image_for_person'](out_size=(384,256)). \
#     inference_onnx_op['resized_image_for_person', 'peson_output'](
#         onnx_path='/workspace/humantracking/coco-person_face_256_384-model.onnx', 
#         mean=[128, 128, 128],
#         std=[128, 128, 128]
#     ). \
#     deploy.detpostprocess_func[('image', 'peson_output'), ('obj_bboxes', 'obj_labels')](level_hw=np.array([32,48,16,24,8,12], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
#     runas_op[('image', 'obj_bboxes', 'obj_labels'), 'person_image'](func=select_person). \
#     resize_op['person_image', 'resized_image_for_seg'](out_size=(384,480)). \
#     inference_onnx_op['resized_image_for_seg', 'seg'](
#         mean=(0.491400*255, 0.482158*255, 0.4465231*255),
#         std=(0.247032*255, 0.243485*255, 0.2615877*255),
#         onnx_path='/workspace/humantracking/personseg-epoch_120-model.onnx',
#     ). \
#     runas_op[('person_image', 'seg'), 'out'](func=debug_show).run()


glob['image_file']('/workspace/dataset/person-seg/sudoku/images/*.png'). \
    image_decode['image_file', 'image'](). \
    resize_op['image', 'resized_image_for_seg'](out_size=(384,480)). \
    inference_onnx_op['resized_image_for_seg', 'seg'](
        mean=(0.491400*255, 0.482158*255, 0.4465231*255),
        std=(0.247032*255, 0.243485*255, 0.2615877*255),
        onnx_path='/workspace/humantracking/personseg-epoch_120-model.onnx',
    ). \
    runas_op[('image', 'seg'), 'out'](func=debug_show).run()