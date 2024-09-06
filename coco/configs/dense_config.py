# 优化器配置
optimizer = dict(type='SGD', lr=0.01,  weight_decay=5e-4, momentum=0.01, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=10))

label_batch_size = 2
unlabel_batch_size = 2

# 学习率调度配置
max_epochs = 80
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 日志配置
log_config = dict(
    interval=1,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 评估方案配置
evaluation=dict(
    out_dir='./output/', 
    interval=1, 
    metric=dict(
        type='COCOCompatibleEval', 
        categories=[{'name': f'{label}', 'id': label} for label in range(2)],
        without_background=False,
        ignore_category_ids=[0]
    ), 
    save_best='AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
    rule='greater'
)

# 模型配置
# model 字段根据具体模型进行设置
model = dict(
    type='DenseTeacher',
    model=dict(
        type='YoloX',
        backbone=dict(
            type='SKetNetF',
            architecture='resnet34',
            in_channels=3,
            out_indices=[2,3,4]
        ),
        neck=dict(type="FPN", in_channels=[96, 128, 160], out_channels=32, num_outs=3),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=2,
            width=1.0,
            strides=[8,16,32],
            in_channels=[32,32,32],
            out_channels=32
        )
    ),
    train_cfg=dict(
        use_sigmoid=True,
        label_batch_size=label_batch_size,
        unlabel_batch_size=unlabel_batch_size,
        semi_ratio=0.5,
        heatmap_n_thr=0.25,
        semi_loss_w=1.0
    ),
    test_cfg=None,
    init_cfg=None,
)

# 自定义 hooks配置
custom_hooks = dict(
    type='MeanTeacher',
    momentum=0.999
)

# 数据配置
data=dict(
    train=[
        dict(
            type="TFDataset",
            data_folder=[
                "/workspace/dataset/beta-badcase-pole-person-ball-priv"
            ],
            pipeline=[
                dict(type='DecodeImage', to_rgb=False),
                dict(type='CorrectBoxes'),
                dict(type='KeepRatio', aspect_ratio=1.7),
                dict(type="ResizeS", target_dim=(384,256)),    # 384,256
                dict(type="Rotation", degree=15),
                dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.4,1.3,0.5], brightness=[0.4,1.3,0.5]),
                dict(type='RandomFlipImage', swap_labels=[]),
                dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
                dict(type='Permute', to_bgr=False, channel_first=True)
            ],
            description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
            inputs_def=dict(
                fields = ["image", 'bboxes', 'labels', 'image_meta']
            ),
            shuffle_queue_size=2048
        ),
        dict(
            type="TFDataset",
            data_folder=[
                "/workspace/dataset/beta-badcase-pole-person-ball-priv"
            ],
            weak_pipeline=[
                dict(type='DecodeImage', to_rgb=False),
                dict(type='CorrectBoxes'),
                dict(type='KeepRatio', aspect_ratio=1.7),
                dict(type="ResizeS", target_dim=(384,256)),    # 384,256
            ],
            strong_pipeline=[
                dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.4,1.3,0.5], brightness=[0.4,1.3,0.5]),
            ],
            pipeline=[
                dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
                dict(type='Permute', to_bgr=False, channel_first=True)        
            ],
            description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
            inputs_def=dict(
                fields = ["image", 'bboxes', 'labels', 'image_meta']
            ),
            shuffle_queue_size=2048
        ),            
    ],
    train_dataloader=dict(
        samples_per_gpu=[label_batch_size, unlabel_batch_size], 
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
        ignore_stack=['image', 'bboxes', 'labels', 'image_meta']
    )
)

