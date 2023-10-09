# 优化器配置
optimizer = dict(type='Adam', lr=0.01,  weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 日志配置
log_config = dict(
    interval=10,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 自定义HOOKS
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

# 模型配置
model = dict(
    type='PoseSegLite',
    backbone=dict(
        type='BackboneDDR',
        architecture='resnet34',
        in_channels=3
    )
)

# 描述模型基本信息
info = dict()

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=[
        dict(
                type='PersonTikTokAndAHPDataset',
                train_or_test='train',
                dir='/dataset',
                pipeline=[
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,256), heatmap_size=(64,64), num_joints=33, sigma=4),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
        dict(
                type='PersonBaiduDataset',
                train_or_test='train',
                dir='/dataset',
                pipeline=[
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.7,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,256), heatmap_size=(64,64), num_joints=33, sigma=4),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            )
    ],
    train_dataloader=dict(
        samples_per_gpu=[64,64], 
        workers_per_gpu=3,
        drop_last=True,
        shuffle=True,
    ),
    test=dict(
        type='PersonBaiduBetaDataset',
        train_or_test='test',
        dir='/workspace/dataset',
        pipeline=[
            dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,256), heatmap_size=(64,64), num_joints=33, with_random=False),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments', 'bboxes'],
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=4, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='OKS', sigma=0.001), save_best='oks', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,256]],
    input_name_list=["image"],
    output_name_list=["heatmap", "offset", "seg"]
)

max_epochs = 80

# deploy=dict(
#     engine='tnn',
#     device='mobile'
# )