# 优化器配置
base_lr = 0.01
optimizer = dict(type='Adam', lr=base_lr,  weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
max_epochs = 120
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
    type='ParsingDDRNetV4',
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
        type='PersonParsingDataset',
        train_or_test='train',
        dir='/workspace/dataset/humanparsing',
        pipeline=[
            dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'parsing_body_segment', 'whole_body_segment', 'has_parsing_body'],
        )),
    dict(
        type='PersonParsingDataset',
        train_or_test='train',
        dir='/workspace/dataset/humanparsing',
        pipeline=[
            dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'parsing_body_segment', 'whole_body_segment', 'has_parsing_body'],
        )),        
    # dict(
    #     type='PersonSegDataset',
    #     train_or_test='train',
    #     dir='/dataset',
    #     pipeline=[
    #         dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
    #         dict(type='ToTensor', keys=['image']),
    #         dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
    #     ],
    #     inputs_def=dict(
    #         fields=['image', 'parsing_body_segment', 'whole_body_segment', 'has_parsing_body'],
    #     )),
    ],
    train_dataloader=dict(
        samples_per_gpu=[1,1],  # 64,64
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
    ),
    test=dict(
        type='PersonParsingTestDataset',
        train_or_test='train',
        dir='/workspace/dataset/humanparsing',
        pipeline=[
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'segments'],
        )),
    test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置(TODO,修改)
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='SegMIOU', class_num=19), save_best='miou', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,256]],
    input_name_list=["image"],
    output_name_list=["seg"],
    deploy=dict(
        engine='rknn',      # rknn,snpe,tensorrt,tnn
        device='rk3588',    # rk3568/rk3588,qualcomm,nvidia,mobile
        preprocess=dict(
            mean_values='125.307,122.95029,113.86339050000001',        # mean values
            std_values='62.99316,62.088675,66.70486349999999'          # std values
        ),
        quantize=False,                 # is quantize
        # calibration=dict(
        #     type='TFDataset',
        #     data_folder = ["/workspace/dataset/person-face"],
        #     pipeline=[
        #         dict(type='DecodeImage', to_rgb=False),
        #         dict(type='KeepRatio', aspect_ratio=1.5),
        #         dict(type="ResizeS", target_dim=(256,192)),
        #     ],
        #     description={'image': 'byte'},
        #     inputs_def=dict(
        #         fields = ["image"]
        #     )
        # ),
        # calibration_size= 1000
    )
)

