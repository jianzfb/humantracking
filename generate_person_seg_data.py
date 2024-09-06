import os
import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/workspace/antgo')
from antgo.pipeline import *
from antgo.dataflow.dataset import *
import random
import uvicorn

# # 创建模板图
# with open('/workspace/dataset/ext/template-for-det.json', 'r') as fp:
#     sample_anno_list = json.load(fp)

# for sample_i in range(len(sample_anno_list)):
#     for index in range(len(sample_anno_list[sample_i]['annotations'][0]['result'])):
#         sample_anno_instance = sample_anno_list[sample_i]['annotations'][0]['result'][index]
#         height = sample_anno_instance['original_height']
#         width = sample_anno_instance['original_width']

#         points = sample_anno_instance['value']['points']
#         label_name = sample_anno_instance['value']['polygonlabels'][0]
#         # label_id = label_name_and_label_id_map[label_name]

#         points_array = np.array(points) 
#         points_array[:, 0] = points_array[:, 0] / 100.0 * width
#         points_array[:, 1] = points_array[:, 1] / 100.0 * height
#         points_array = points_array.astype(np.int32)
#         name = sample_anno_list[sample_i]['file_upload'].split('/')[-1][9:]

#         image = cv2.imread(f'/workspace/dataset/ext/template-for-det/{name}')
#         mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#         mask = cv2.fillPoly(mask, [points_array], 255)
#         cv2.imwrite(f'/workspace/dataset/ext/template-for-det-rgba/{name}', np.concatenate([image, np.expand_dims(mask, -1)], -1))

# LayoutTemplateGenerator 3种加载方式，label-studio, 本地目录

# # 座位体前屈分割
# glob['image_path']('/workspace/dataset/ext/ball-ext-dataset/background/spider_zhanhui/test/*', repeat=50). \
#     image_decode['image_path', 'image'](). \
#     sync_layout_op['image', 'layout-1'](
#         layout_gen=[
#             LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg-rgba/', min_scale=0.5, max_scale=0.7, keep_prefix='device'), 
#             LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg-rgba/', min_scale=0.9, max_scale=1.0, ignore_prefix='device')
#         ],
#         layout_id=[1,2]). \
#     sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.7, max_scale=0.9, keep_layout=[2], layout_label_map={2:1}). \
#     select['sync-out']().as_raw(). \
#     to_dataset('./my', is_tfrecord=True)


# 座位体前屈分割
glob['image_path']('/workspace/dataset/ext/ball-ext-dataset/background/spider_zhanhui/test/*', repeat=25). \
    image_decode['image_path', 'image'](). \
    sync_layout_op['image', 'layout-1'](
        layout_gen=[
            LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg.json', data_folder='/workspace/dataset/ext/template-for-seg', min_scale=0.5, max_scale=0.7, keep_prefix='qicai'), 
            LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg.json', data_folder='/workspace/dataset/ext/template-for-seg', min_scale=0.9, max_scale=1.0, ignore_prefix='qicai')
        ],
        layout_id=[1,2]). \
    sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.7, max_scale=0.9, keep_layout=[2], layout_label_map={2:1}). \
    select['sync-out']().as_raw(). \
    to_dataset('/workspace/dataset/ext/tiqianqu-ext-dataset', is_tfrecord=False)


# # 目标检测
# glob['image_path']('/workspace/dataset/ext/ball-ext-dataset/background/spider_zhanhui/test/*', repeat=25). \
#     image_decode['image_path', 'image'](). \
#     sync_layout_op['image', 'layout-1'](
#         layout_gen=[
#             LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg-rgba/', min_scale=0.8, max_scale=0.9, keep_prefix='device'), 
#             LayoutTemplateGenerator('/workspace/dataset/ext/template-for-det-rgba/', min_scale=0.9, max_scale=1.0, ignore_prefix='device')
#         ],
#         layout_id=[1,2]). \
#     sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.7, max_scale=0.9, keep_layout=[2], layout_label_map={2:0}). \
#     select['sync-out']().as_raw(). \
#     to_dataset('/workspace/dataset/ext/person-sitpose-ext-dataset', is_tfrecord=True, keymap={'image': 'image', 'labels': 'labels', 'bboxes': 'bboxes'})


# def generate_sync_data_pipeline():
#     background_path = ''
#     template_path='' 
#     sync_folder =''
#     glob['image_path'](f'{background_path}/*', repeat=10). \
#         image_decode['image_path', 'image'](). \
#         sync_layout_op['image', 'layout-1'](
#             layout_gen=[
#                 LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg-rgba/', min_scale=0.5, max_scale=0.7), 
#             ], 
#             layout_id=[1]). \
#         sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.7, max_scale=0.9). \
#         select['sync-out']().as_raw(). \
#         to_dataset(sync_folder, is_tfrecord=True)

# image_package 完成图像信息打包(所有样本)，复制到目标目录，并生成json
# with web['search'](name='step 0', step_i=0, step_num=2) as handler:
#     app = handler.image_json_record['search', ('sample_num', 'sample_list', 'record_file')](folder='./my/background', prefix='background'). \
#         demo(
#             title="收集底层图",
#             description="收集底层图", 
#             input=[
#                 {'data': 'search', 'type': 'image-search'},
#             ], 
#             output=[
#                 {'data': 'sample_num', 'type': 'text'}
#             ]
#         )

# # image_record 完成图像记录（每一一个样本），返回总数和列表， 并生成json
# with web['upload'](name='step 1', step_i=1, step_num=2) as handler:
#     app = handler.interactive_polygon('upload', 'polygons', num=1). \
#         fillpoly[('upload', 'polygons'), 'template'](fill=1, is_overlap=False). \
#         image_json_record['template', ('template_num', 'template_list', 'record_file')](folder='./my/template', prefix='template'). \
#         demo(
#             title="标注模板图",
#             description="标注模板图", 
#             input=[
#                 {'data': 'upload', 'type': 'image'},
#             ], 
#             output=[
#                 {'data': 'template_num', 'type': 'text'},
#             ]
#         )


# def generate_sync_data_func():
#     # 座位体前屈分割
#     glob['image_path']('/workspace/dataset/ext/ball-ext-dataset/background/spider_zhanhui/test/*', repeat=1). \
#         image_decode['image_path', 'image'](). \
#         sync_layout_op['image', 'layout-1'](
#             layout_gen=[
#                 LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg.json', data_folder='/workspace/dataset/ext/template-for-seg', min_scale=0.5, max_scale=0.7, keep_prefix='qicai'), 
#                 LayoutTemplateGenerator('/workspace/dataset/ext/template-for-seg.json', data_folder='/workspace/dataset/ext/template-for-seg', min_scale=0.9, max_scale=1.0, ignore_prefix='qicai')
#             ],
#             layout_id=[1,2]). \
#         sync_op[('image', 'layout-1'), 'sync-out'](min_scale=0.7, max_scale=0.9, keep_layout=[2], layout_label_map={2:1}). \
#         select['sync-out']().as_raw(). \
#         to_dataset('./my', is_tfrecord=True)
#     return 10


# with web(name='generate') as handler:
#     app = handler.command_op['out'](func=generate_sync_data_func). \
#         demo(
#             title="生成",
#             description="生成", 
#             output=[
#                 {'data': 'out', 'type': 'text'},
#             ]
#         )


# if __name__ == "__main__":
# 	uvicorn.run(app, host="0.0.0.0", port=8000)