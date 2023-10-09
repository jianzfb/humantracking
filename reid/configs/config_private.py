# 优化器配置
optimizer = dict(
    type='SGD', 
    lr=0.05,  
    weight_decay=5e-4, 
    momentum=0.9, 
    nesterov=True,
    paramwise_cfg = dict(
        custom_keys={
            'model': dict(lr_mult=0.1),
            'classifier': dict(lr_mult=1.0),
        }
    )
)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='Step',
    step=60*2//3,
    gamma=0.1
)


# 日志配置
log_config = dict(
    interval=50,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
model = dict(
    type='ReidClassifier',
    train_cfg=dict(
        class_num=4971,
        droprate=0.5,
        stride=2,
        circle=True,
        linear_num=512
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type="ImageFolder",
        root="/dataset/reid-private/pytorch/train",
        pipeline=[
            dict(type='Resize', size=(256,128), interpolation=3, keys=['image']),
            dict(type='Pad', padding=10, keys=['image']),
            dict(type='RandomCrop', size=(256,128), keys=['image']),
            dict(type='RandomHorizontalFlip', keys=['image']),
            dict(type='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=0, keys=['image']),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], keys=['image'])
        ],
        inputs_def={'fields': ['image', 'label'], 'alias': ['image', 'label']},
    ),
    train_dataloader=dict(
        samples_per_gpu=64, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
    ),    
    test=dict(
        type="MSMTDataset",
        dir="/workspace/dataset/MSMT17_V1/pytorch",
        train_or_test='test',
        pipeline=[    
            dict(type='Resize', size=(256,128), interpolation=3, keys=['image']),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], keys=['image'])            
        ],
        inputs_def={'fields': ['image', 'label', 'tag', 'camera'], 'alias': ['image', 'label', 'tag', 'camera']},
    ),
    test_dataloader=dict(
        samples_per_gpu=16, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='ReidEval'), save_best='Rank@1')

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,128]],
    input_name_list=["image"],
    output_name_list=["feature"]
)

max_epochs = 60