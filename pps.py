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

# video_dc['image']('/workspace/project/sports/volleyball/9_20230531_1_24.mp4'). \
#     select('out').as_raw().to_video(output_path='./aa.mp4', width=1920, height=1080)
# glob['file_path']('/workspace/dataset/test/*.png'). \
#     image_decode['file_path', 'image'](). \
# video_dc['image', 'frame_index']('/workspace/humantracking/badcase1.mp4'). \
#     keep_ratio_op['image', 'keep_ratio_image'](aspect_ratio=1.77). \
#     resize_op['keep_ratio_image', 'resized_image'](out_size=(704,384)). \
#     inference_onnx_op['resized_image', 'output'](
#         onnx_path='/workspace/humantracking/coco-epoch_40-model.onnx', 
#         mean=[128, 128, 128],
#         std=[128, 128, 128]
#     ). \
#     runas_op['output', ('box', 'label')](func=post_process_func).\
#     plot_bbox[("resized_image", "box", 'label'),"out"](thres=0.1, color=[[0,0,255], [0,255,0]], category_map={'0': 'person', '1': 'ball'}). \
#     image_save['out', 'save'](folder='./CC/').run()

skeleton = [
    [0,9],
    [0,10],
    [9,10],
    [11,12],
    [11,13],
    [12,14],
    [11,23],
    [12,24],
    [23,24],
    [23,25],
    [24,26],
    [25,27],
    [26,28],
    [27,29],
    [27,31],
    [31,29],
    [28,32],
    [28,30],
    [30,32],
    [13,15],
    [14,16]
]

def pose_draw(image, heatmap, offset):
    image_h, image_w = image.shape[:2]
    heatmap_h, heatmap_w = heatmap.shape[2:]

    pose_points = []
    num_joints = 33
    for channel_i in range(num_joints):
        aa = np.argmax(heatmap[0,channel_i])
        r = aa // heatmap_w
        c = aa % heatmap_w
        if heatmap[0,channel_i,r,c] < 0.2:
            pose_points.append((-1,-1))
            continue

        offset_x = offset[0, channel_i, r, c]
        offset_y = offset[0, num_joints+channel_i, r, c]

        point_x = c + offset_x
        point_y = r + offset_y

        point_x = point_x * (image_w/heatmap_w)
        point_y = point_y * (image_h/heatmap_h)
        image = cv2.circle(image, (int(point_x), int(point_y)), 1, (0,0,255),1)
        pose_points.append((point_x, point_y))
    
    for s,e in skeleton:
        start_x,start_y = pose_points[s]
        end_x, end_y = pose_points[e]
        if start_x < 0 or end_x < 0:
            continue

        cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,0,0), 1)
    return image


def debug_show(frame_index, resized_image, heatmap, offset):
    image_h, image_w = resized_image.shape[:2]

    canvas = np.zeros((image_h, image_w*3,3), dtype=np.uint8)
    # cv2.imwrite(f'./temp/a_{frame_index}.png', seg_map)
    # cv2.imwrite(f'./temp/b_{frame_index}.png', resized_image)
    canvas[:,:image_w] = resized_image
    resized_image = pose_draw(resized_image, heatmap, offset)
    canvas[:,image_w:image_w*2] = resized_image
    # cv2.imwrite(f'./temp/c_{frame_index}.png', resized_image)
    # canvas[:,image_w*2:image_w*3] = np.stack([seg_map, seg_map, seg_map], -1)

    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    # frame_index = 0
    cv2.imwrite(f'./temp/canvas_{frame_index}.png', canvas)
    print('./')


def select_person_func(image, person_bboxes, person_labels):
    selected_i = person_labels == 0
    selected_person_bboxes = person_bboxes[selected_i]
    x0,y0,x1,y1,score = selected_person_bboxes[0]
    image_h, image_w = image.shape[:2]
    x0 = max(int(x0-30),0)
    y0 = max(int(y0-30),0)
    x1 = min(int(x1+30), image_w)
    y1 = min(int(y1+30), image_h)
    person_image = image[y0:y1,x0:x1]
    return person_image


def gg_func(image):
    cv2.imwrite('./aa.png', image)


op.load('detpostprocessop', '/workspace/project/sports/volleyball')
# glob['file_path']('/workspace/dataset/mm/dataset/tiaoyuan_refine/images/2_20230531/*.jpg'). \
#     image_decode['file_path', 'image'](). \
video_dc['image', 'frame_index']('/workspace/dataset/video/12_26_1693903897.mp4'). \
    keep_ratio_op['image', ('keep_ratio_image_for_det', 'keep_ratio_bbox')](aspect_ratio=1.77). \
    resize_op['keep_ratio_image_for_det', 'resized_image_for_deg'](out_size=(704,384)). \
    runas_op['resized_image_for_deg', 'mm'](func=gg_func). \
    inference_onnx_op['resized_image_for_deg', 'person_det'](
        mean=(128, 128, 128),
        std=(128, 128, 128),
        onnx_path='/workspace/humantracking/coco-epoch_20_coco-fix-model.onnx',
    ). \
    deploy.detpostprocess_func[('image','person_det'), ('person_bboxes', 'person_labels')](level_hw=np.array([48,88,24,44,12,22], dtype=np.int32), level_strides=np.array([8,16,32], dtype=np.int32)). \
    runas_op[('image', 'person_bboxes', 'person_labels'), 'person_image'](func=select_person_func). \
    keep_ratio_op['person_image', ('keep_ratio_image_for_pose', 'keep_ratio_bbox')](aspect_ratio=0.75). \
    resize_op['keep_ratio_image_for_pose', 'resized_image_for_pose'](out_size=(192,256)). \
    inference_onnx_op['resized_image_for_pose', ('heatmap', 'offset')](
        onnx_path='/workspace/humantracking/poseseg-epoch_270-model.onnx', 
        mean=[0.491400*255, 0.482158*255, 0.4465231*255],
        std=[0.247032*255, 0.243485*255, 0.2615877*255],
    ). \
    runas_op[('frame_index','resized_image_for_pose', 'heatmap', 'offset'), 'out'](func=debug_show).run()