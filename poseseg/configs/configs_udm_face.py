import numpy as np

project_name = "humantrackingface"

# 优化器配置
base_lr = 0.005
optimizer = dict(
    type='AdamW', lr=base_lr,  weight_decay=0.05,
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0, 
        bypass_duplicate=True
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

max_epochs = 130
# 学习率调度配置
lr_config = [
    dict(
        policy='Warmup',
        begin=0,
        end=5,
        warmup_iters=5,
        warmup='linear',
        warmup_by_epoch=False
    ),
    dict(
        policy='CosineAnnealing',
        min_lr=0.0005,
        by_epoch=True,
        begin=5,
        end=120,
    ),
    dict(policy='Fixed', by_epoch=True, factor=1, begin=120, end=130),
]
# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# 日志配置
log_config = dict(
    interval=10,         # 10
    hooks=[
        dict(
            type='VibSLoggerHook', 
            record_keys=['lr', 'layout_1_loss_uv_hm', 'layout_1_loss_xy_offset', 'loss']
        ),
    ]
)

# 自定义HOOKS
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

# 模型配置
model = dict(
    type='PoseDDRNetV4',
    backbone=dict(
        type='UDM', 
        in_channels=3,
        mid_channels=64, 
        out_channels=64,
        pan_mode=False
    )
)

# checkpoint配置
checkpoint_config = dict(interval=5, out_dir='./output/')       

num_keypoints = 11
cocowholebody_betamp33 = [
    (0, 0), (1, 2), (2, 5), (3, 7), (4, 8),  
    (59, 6), (62, 4), (65, 1), (68, 3),
    (71, 10), (77, 9)
]

halpe_betamp33 = [(0, 0), (1, 2), (2, 5), (3, 7), (4, 8), 
				  (62, 6), (65, 4), (68, 1), (71, 3), 
				  (74, 10), (80, 9)]


# 数据配置
data=dict(
        train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoWholeBodyDataset',
                dir='/dataset/human_keypoints/coco-wholebody',
                ann_file='annotations/coco_wholebody_train_v1.0.json',
                data_prefix=dict(img='train2017/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=cocowholebody_betamp33),
                ]
            ),
            dict(
                type='HalpeDataset',
                dir='/dataset/human_keypoints/halpe',
                ann_file='annotations/halpe_train_v1.json',
                data_prefix=dict(img='hico_20160224_det/images/train2015/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=halpe_betamp33),
                ]
            ),
        ],
        pipeline=[
            dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
            dict(type='ColorDistort', hue=[-10,10,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
            dict(
                type='RandomFlipImage', 
                swap_ids=[[
                    4,5,6,8,10
                ],[
                    1,2,3,7,9
                ]], 
                swap_labels=[]
            ),
            dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,256), heatmap_size=(64,64), num_joints=11, sigma=2, scale_factor=0.12,center_factor=0.05),
            dict(type="AddExtInfo"),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'has_joints'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=64, 
        workers_per_gpu=4,
        drop_last=True,
        shuffle=True,
    )
)

# 评估方案配置
keypoint_weights=[
    1., 1., 1., 1., 1., 1., 1., 1.,1.,1.,1.
]
keypoint_sigmas=[
    0.026, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.035, 0.035, 0.035, 0.035
]

evaluation = [
    # 评估关键点
    dict(out_dir='./output/', interval=1, metric=dict(type='OKS', sigmas=keypoint_sigmas, weights=keypoint_weights), save_best='oks', rule='greater'),
    # 评估分割
    dict(out_dir='./output/', interval=1, metric=dict(type='SegMIOU', class_num=2), save_best='miou', rule='greater')
]

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,256]],
    input_name_list=["image"],
    output_name_list=["heatmap", "offset"],
)