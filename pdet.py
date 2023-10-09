import enum
import sys
sys.path.insert(0,'/workspace/antgo')

from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
import cv2
import numpy as np


def nms(boxes, labels, iou_thres):
    """ 非极大值抑制 """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2-x1) * (y2-y1)
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while(index.size):
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])

        if(index.size == 1): # 如果只剩一个框，直接返回
            break

        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thres)[0]
        index = index[ids+1]

    return boxes[keep], labels[keep]


def post_process_func(
    heatmap_level_1, heatmap_level_2, heatmap_level_3, 
    offset_level_1, offset_level_2, offset_level_3):
    level_stride_list = [8,16,32]
    level_heatmap_list = [heatmap_level_1, heatmap_level_2, heatmap_level_3]
    level_offset_list = [offset_level_1, offset_level_2, offset_level_3]

    label_bboxes = []
    label_labels = []
    for label in range(2):
        all_bboxes = []
        all_labels = []
        for level_i, level_stride in enumerate(level_stride_list):
            level_local_cls = level_heatmap_list[level_i][0, label]  # 仅抽取person cls channel
            level_offset = level_offset_list[level_i]

            height, width = level_local_cls.shape
            flatten_local_cls = level_local_cls.flatten()
            topk_inds = np.argsort(flatten_local_cls)[::-1][:100]
            topk_scores = flatten_local_cls[topk_inds]
            pos = np.where(topk_scores > 0.6)
            if pos[0].size == 0:
                continue

            topk_inds = topk_inds[pos]
            topk_scores = flatten_local_cls[topk_inds]

            topk_inds = topk_inds % (height * width)
            topk_ys = (topk_inds // width).astype(np.float32)
            topk_xs = (topk_inds % width).astype(np.float32)

            local_reg = np.transpose(level_offset, [0,2,3,1])   # BxHxWx4
            local_reg = np.reshape(local_reg, [-1,4])
            topk_ltrb_off = local_reg[topk_inds]

            tl_x = (topk_xs * level_stride + level_stride//2 - topk_ltrb_off[:,0] * level_stride)
            tl_y = (topk_ys * level_stride + level_stride//2 - topk_ltrb_off[:,1] * level_stride)
            br_x = (topk_xs * level_stride + level_stride//2 + topk_ltrb_off[:,2] * level_stride)
            br_y = (topk_ys * level_stride + level_stride//2 + topk_ltrb_off[:,3] * level_stride)

            bboxes = np.stack([tl_x,tl_y,br_x,br_y, topk_scores], -1)
            labels =  np.array([label]*bboxes.shape[0])
            all_bboxes.append(bboxes)
            all_labels.append(labels)

        if len(all_bboxes) == 0:
            continue

        all_bboxes = np.concatenate(all_bboxes, 0)
        all_labels = np.concatenate(all_labels)
        if label == 1:
            all_bboxes, all_labels = nms(all_bboxes, all_labels, 0.01)
        else:
            all_bboxes, all_labels = nms(all_bboxes, all_labels, 0.1)

        label_bboxes.extend(all_bboxes.tolist())
        label_labels.extend(all_labels.tolist())

    return np.array(label_bboxes), np.array(label_labels)

# video_dc['image']('/workspace/project/sports/volleyball/9_20230531_1_24.mp4'). \
#     select('out').as_raw().to_video(output_path='./aa.mp4', width=1920, height=1080)
glob['file_path']('/workspace/dataset/mm/dataset/zq_dataset/train/image/*.jpg'). \
    image_decode['file_path', 'image'](). \
    keep_ratio_op['image', 'keep_ratio_image'](aspect_ratio=1.77). \
    resize_op['keep_ratio_image', 'resized_image'](out_size=(704,384)). \
    inference_onnx_op['resized_image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')](
        onnx_path='/workspace/humantracking/coco-epoch_100_new_v2-model.onnx', 
        mean=[128, 128, 128],
        std=[128, 128, 128]
    ). \
    runas_op[('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3'), ('box', 'label')](func=post_process_func).\
    plot_bbox[("resized_image", "box", 'label'),"out"](thres=0.2, color=[[0,0,255], [0,255,0]], category_map={'0': 'person', '1': 'ball'}). \
    image_save['out', 'save'](folder='./CC/').run()

