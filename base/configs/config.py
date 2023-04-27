# 优化器配置
optimizer = dict(type='Adam', lr=0.00005,  weight_decay=0)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='Step',
    step=[1],
    min_lr=1e-6
)

# 日志配置
log_config = dict(
    interval=50,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# # 模型配置
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='WideResNet',
#         num_classes=10,     # TODO 28
#         depth=28,
#         widen_factor=2,     # TODO 8
#         dropout=0,
#         dense_dropout=0.2          
#     ),
#     head=dict(
#         type='ClsHead'
#     )
# )
model = dict(
    type="PyMAF",
    train_cfg=dict(
        MAF_ON= False,
        BACKBONE= 'res50',
        MLP_DIM= [256, 128, 64, 5],
        N_ITER= 3,
        AUX_SUPV_ON= True,
        DP_HEATMAP_SIZE= 56,        
        RES_MODEL=dict(
            DECONV_WITH_BIAS= False,
            NUM_DECONV_LAYERS= 3,
            NUM_DECONV_FILTERS=[256,256,256],
            NUM_DECONV_KERNELS=[4,4,4]      
        ),
        POSE_RES_MODEL=dict(
            INIT_WEIGHTS= True,
            NAME= 'pose_resnet',
            PRETR_SET= 'imagenet',   # 'none' 'imagenet' 'coco'
            # PRETRAINED: 'data/pretrained_model/resnet50-19c8e357.pth'
            PRETRAINED_IM= 'data/pretrained_model/resnet50-19c8e357.pth',
            PRETRAINED_COCO= 'data/pretrained_model/pose_resnet_50_256x192.pth.tar',
            EXTRA=dict(
                TARGET_TYPE= 'gaussian',
                HEATMAP_SIZE=[48,64],
                SIGMA= 2,
                FINAL_CONV_KERNEL= 1,
                DECONV_WITH_BIAS= False,
                NUM_DECONV_LAYERS= 3,
                NUM_DECONV_FILTERS=[256,256,256],
                NUM_DECONV_KERNELS=[4,4,4],
                NUM_LAYERS= 50   
            )
        )
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type="TFDataset",
        data_path_list=[],  # 'ali:///dataset/cifar10/cifar_10_train-00000-of-00001'
        pipeline=[    
            dict(type='Meta', keys=['image_file', 'tag']),
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),
            dict(type='RandomHorizontalFlip', keys=['image']),
            dict(type='RandomCrop', size=(32,32), padding=int(32*0.125), fill=128, padding_mode='constant', keys=['image']), 
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label', 'image_meta']},
        description={'image': 'numpy', 'label': 'int', 'image_file': 'str', 'tag': 'str'}
    ),
    train_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
    ),    
    val=dict(
        type="TFDataset",
        data_path_list=[],  # 'ali:///dataset/cifar10/cifar_10_test-00000-of-00001'
        pipeline=[    
            dict(type='Meta', keys=['image_file', 'tag']),
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),                  
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),                  
        ],
        inputs_def={'fields': ['image', 'label', 'image_meta']},
        description={'image': 'numpy', 'label': 'int', 'image_file': 'str', 'tag': 'str'}
    ),
    val_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    ),   
    test=dict(
        type="TFDataset",
        data_path_list=[],  # 'ali:///dataset/cifar10/cifar_10_test-00000-of-00001'
        pipeline=[    
            dict(type='Meta', keys=['image_file', 'tag']),
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),                  
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),                  
        ],
        inputs_def={'fields': ['image', 'label', 'image_meta']},
        description={'image': 'numpy', 'label': 'int', 'image_file': 'str', 'tag': 'str'},
    ),
    test_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    ),       
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='AccuracyEval'), save_best='top_1')

# 导出配置
export=dict(
    input_shape_list = [[1,3,32,32]],
    input_name_list=["image"],
    output_name_list=["pred"]
)

max_epochs = 60