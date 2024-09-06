# 优化器配置
base_lr = 0.002
optimizer = dict(
    type='AdamW', lr=base_lr,  weight_decay=0.05,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)
optimizer_config = dict(grad_clip=None)

max_epochs = 120

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr*0.05,
    by_epoch=True,
    begin=20,
    end=max_epochs,
    warmup_iters=1000,
    warmup='linear',
    warmup_by_epoch = False
)

# 自定义HOOKS
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# 日志配置
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
model = dict(
    type='PersonSegDDR',
    backbone=dict(
        type='BackboneDDR',
        architecture='resnet34',
        in_channels=3
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type='PersonSegDataset',
        train_or_test='train',
        dir='/dataset',
        pipeline=[
            dict(type='KeepRatio', aspect_ratio=0.8),
            dict(type='ResizeS', target_dim=(420,512)),
            dict(type='Rotation', degree=30),
            dict(type='RandomCropImageV1', size=(384, 480), padding=0, fill=0, prob=1.0),
            dict(type='ColorDistort'),
            dict(type='RandomFlipImage', swap_ids=[], swap_labels=[]),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            dict(type='UnSqueeze', axis=0, keys=['segments'])
        ],
        inputs_def=dict(
            fields=['image', 'segments'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=96, 
        workers_per_gpu=3,
        drop_last=True,
        shuffle=True,
    ),
    test=dict(
        type='Pascal2012',
        train_or_test='val',
        dir='./pascal2012_dataset',
        ext_params=dict(task_type='SEGMENTATION', aug=True),
        pipeline=[
            dict(type='ResizeS', target_dim=(256,256)),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            dict(type='UnSqueeze', axis=0, keys=['segments'])
        ],                
        inputs_def=dict(
            fields=['image', 'segments'],
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='SegMIOU', class_num=1), save_best='miou', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,480,384]],
    input_name_list=["image"],
    output_name_list=["seg"]
)