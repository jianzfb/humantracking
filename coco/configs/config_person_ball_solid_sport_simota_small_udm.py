base_lr = 0.007
# 优化器配置
optimizer = dict(type='Adam', lr=base_lr,  weight_decay=5e-4)  # 0.01
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
max_epochs = 120
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr*0.05,
    by_epoch=True,
    begin=max_epochs//2,
    end=max_epochs,
    warmup_iters=1000,
    warmup='linear',
    warmup_by_epoch = False
)

# 日志配置
log_config = dict(
    interval=10,         # 10
    hooks=[
        dict(
            type='VibSLoggerHook', 
            record_keys=['lr', 'loss_iou', 'loss_obj', 'loss_cls', 'epoch']
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
    type='YoloX',
    backbone=dict(
        type='UDMLite',
        in_channels=3,
        mid_channels=32, 
        out_channels=32,
        out_strides=[8,16,32]
    ),
    neck=None,
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=2,
        width=1.0,
        strides=[8,16,32],
        in_channels=[32,32,32],
        out_channels=32
    )
)

# checkpoint配置
checkpoint_config = dict(interval=5, out_dir='./output/')

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# 数据配置
data=dict(
    train=dict(
        type='TFDataset',
        data_folder = [
            "/dataset/personball/large-humanbody-ball-priv",
            "/dataset/personball/beta-humanbody-ball-priv",
            "/dataset/personball/beta-humanbody-shixin-ball-filter-priv",
            "/dataset/personball/beta-sync-ball-v3-priv",
            "/dataset/personball/beta-football-3rd-priv",
            "/dataset/personball/beta-sync-person-ball-v2-priv",
            "/dataset/personball/beta-badcase-invlight-sync-person-ball-priv",
            "/dataset/personball/beta-badcase-invlight-sync-person-ball-v2-priv",
            "/dataset/personball/beta-sync-person-ball-v3-priv",
            "/dataset/personball/beta-badcase-pole-person-ball-priv",
            "/dataset/personball/sync_ext_zhanhui",
            "/dataset/personball/real_ext_crop_zhanhui",
            "/dataset/personball/real_ext_crop_zhanhui",
            "/dataset/personball/real_ext_crop_zhanhui",
            "/dataset/personball/real_ext_zhanhui",
            "/dataset/personball/real_ext_zhanhui",
            "/dataset/personball/real_ext_zhanhui",
            "/dataset/personball/zhanhuixianchang_ext_zhanhui",
            "/dataset/personball/zhanhuixianchang_ext_zhanhui",
            "/dataset/personball/zhanhuixianchang_ext_zhanhui",
        ],
        pipeline=[
            dict(type='DecodeImage', to_rgb=False),
            dict(type='CorrectBoxes'),
            dict(type='KeepRatio', aspect_ratio=1.7),
            dict(type="ResizeS", target_dim=(384,256)),    # 384,256
            dict(type="Rotation", degree=15),
            dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.4,1.3,0.5], brightness=[0.4,1.3,0.5]),
            dict(type='RandomFlipImage', swap_labels=[]),
            dict(type="YoloBboxFormat"),
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='Permute', to_bgr=False, channel_first=True)
        ],
        description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
        inputs_def=dict(
            fields = ["image", 'bboxes', 'image_meta']
        ),
        shuffle_queue_size=4096
    ),
    train_dataloader=dict(
        samples_per_gpu=64,         # 64
        workers_per_gpu=4,          # 4
        drop_last=True,
        shuffle=True,
        ignore_stack=['image', 'bboxes', 'image_meta']
    ),
    test=dict(
        type='TFDataset',
        data_folder = [
            "/workspace/dataset/personball-public/test", 
        ],
        pipeline=[
            dict(type='DecodeImage', to_rgb=False),
            dict(type='CorrectBoxes'),
            dict(type='KeepRatio', aspect_ratio=1.77, focus_on_center=True),
            dict(type="ResizeS", target_dim=(384,256)),    # 384,256
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='Permute', to_bgr=False, channel_first=True)
        ],
        description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
        inputs_def=dict(
            fields = ["image", 'image_meta', 'raw_ann_info']
        ),
        shuffle_queue_size=256
    ),
    test_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
        ignore_stack=['image', 'bboxes', 'labels', 'image_meta']
    )
)

# 评估方案配置
evaluation=dict(
    out_dir='./output/', 
    interval=1, 
    metric=dict(
        type='COCOCompatibleEval', 
        categories=[{'name': f'{label}', 'id': label} for label in range(2)],
        without_background=False
    ), 
    save_best='AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
    rule='greater'
)

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,384]],
    input_name_list=["image"],
    output_name_list=["output"]
)

seed=12