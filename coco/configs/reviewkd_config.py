# 优化器配置
optimizer = dict(type='SGD', lr=0.01,  weight_decay=0.0001, momentum=0.9, nesterov=False)
optimizer_config = dict(grad_clip=None)


# 学习率调度配置
max_epochs = 80
lr_config = dict(
    policy='Step',
    step=[40,60],
    gamma=0.1,
    warmup_iters=1000,
    warmup='linear',
    warmup_by_epoch = False    
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
    ), 
    save_best='AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
    rule='greater'
)

# 模型配置
# model 字段根据具体模型进行设置
model = dict(
    type='ReviewKD',
    model=dict(
        teacher=dict(
            type='YoloX',
            backbone=dict(
                type='ResnetTV',
                model='resnet50',
                pretrained=True,
                output=[1,2,3]
            ),
            neck=dict(type="FPN", in_channels=[512, 1024, 2048], out_channels=32, num_outs=3),
            bbox_head=dict(
                type='YOLOXHead',
                num_classes=2,
                width=1.0,
                strides=[8,16,32],
                in_channels=[32,32,32],
                out_channels=32
            )
        ),
        student=dict(
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
        )
    ),
    train_cfg=dict(
        student=dict(
            channels=[96,128,160],
        ),
        teacher=dict(
            channels=[512, 1024, 2048],
        ),
        kd_loss_weight=1.0,
        kd_warm_up=1,
        align = 'backbone',
        ignore = ['neck', 'bbox_head']
    ),
    test_cfg=None,
    init_cfg=None,
)

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# 数据配置
data=dict(
    train=dict(
        type='TFDataset',
        data_folder = [
            # "/dataset/large-humanbody-ball-priv",
            # "/dataset/beta-humanbody-ball-priv",
            # "/dataset/beta-humanbody-shixin-ball-filter-priv",
            # "/dataset/beta-sync-ball-v3-priv",
            # "/dataset/beta-football-3rd-priv",
            # "/dataset/beta-sync-person-ball-v2-priv",
            # "/dataset/beta-badcase-invlight-sync-person-ball-priv",
            # "/dataset/beta-badcase-invlight-sync-person-ball-v2-priv",
            # "/dataset/beta-sync-person-ball-v3-priv",
            "/workspace/dataset/beta-badcase-pole-person-ball-priv"
        ],
        pipeline=[
            dict(type='DecodeImage', to_rgb=False),
            dict(type='CorrectBoxes'),
            dict(type='KeepRatio', aspect_ratio=1.7),
            dict(type="ResizeS", target_dim=(384,256)),    # 384,256
            dict(type="Rotation", degree=20),
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
        samples_per_gpu=2,          # 128
        workers_per_gpu=1,          # 4
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
            fields = ["image", 'bboxes', 'labels', 'image_meta']
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

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,384]],
    input_name_list=["image"],
    output_name_list=["output"]
)