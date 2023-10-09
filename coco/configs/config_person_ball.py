# 优化器配置
optimizer = dict(type='Adam', lr=0.01,  weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup_by_epoch=False,
    warmup_iters=2000,
    warmup='linear'
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
    type='TTFNet',
    backbone=dict(
        type='SKetNetF',
        architecture='resnet34',
        in_channels=3,
        out_indices=[2,3,4]
    ),     
    neck=dict(type="FPN", in_channels=[96, 128, 160], out_channels=32, num_outs=3),
    bbox_head=dict(
        type='FcosHeadML',
        in_channel=32,
        feat_channel=32,
        num_classes=2,
        down_stride=[8,16,32],
        score_thresh=0.05,
        train_cfg=dict(
            limit_range={8:[-1, 48], 16: [48, 128], 32: [128, 99999]}),
        test_cfg=dict(topk=100, local_maximum_kernel=3, nms=0.6, max_per_img=50),
        loss_ch=dict(type='GaussianFocalLoss', loss_weight=3.0),
        loss_rg=dict(
            type='IouLoss', loss_weight=0.5),
        init_cfg=[
                dict(type='Kaiming', layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]
    ),
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type='TFDataset',
        data_folder = ["/dataset/large-humanbody-ball-priv", "/dataset/beta-humanbody-ball-priv", "/dataset/beta-humanbody-shixin-ball-filter-priv"],
        pipeline=[
                dict(type='DecodeImage', to_rgb=False),
                dict(type='CorrectBoxes'),
                dict(type='KeepRatio', aspect_ratio=1.77),
                dict(type="ResizeS", target_dim=(704, 384)),
                dict(type="Rotation", degree=30),
                dict(type='RandomCropImageV1', size=(704,384), padding=40, fill=128),
                dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.7,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.7,1.3,0.5]),
                dict(type='RandomFlipImage', swap_labels=[]),
                dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
                dict(type='Permute', to_bgr=False, channel_first=True)
            ],
        description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
        inputs_def=dict(
            fields = ["image", 'bboxes', 'labels', 'image_meta']
        ),
        shuffle_queue_size=4096
    ),
    train_dataloader=dict(
        samples_per_gpu=128,
        workers_per_gpu=4,
        drop_last=True,
        shuffle=True,
        ignore_stack=['image', 'bboxes', 'labels', 'image_meta']
    )
)

# 评估方案配置
evaluation=dict(
    out_dir='./output/', 
    interval=1, 
    metric=dict(
        type='COCOCompatibleEval', 
        categories=[{'name': f'{label}', 'id': label} for label in range(80)],
        without_background=False
    ), 
    save_best='AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
    rule='greater'
)

# 导出配置
export=dict(
    input_shape_list = [[1,3,384,704]],
    input_name_list=["image"],
    output_name_list=["heatmap_level_1", "heatmap_level_2", "heatmap_level_3", "offset_level_1", "offset_level_2", "offset_level_3"]
)

max_epochs = 100