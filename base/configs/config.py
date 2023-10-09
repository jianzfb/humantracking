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

# 模型配置
model = dict(
    type="PyMAF",
    train_cfg=dict(
        MAF_ON= False,
        BACKBONE= 'res50',
        MLP_DIM= [256, 128, 64, 5],
        N_ITER= 3,
        AUX_SUPV_ON= False,
        DP_HEATMAP_SIZE= 56,
        batch_size=2,
        LOSS=dict(
            KP_2D_W=300.0,
            KP_3D_W=300.0,
            SHAPE_W=0.06,
            POSE_W=60.0,
            VERT_W=0.0,
            INDEX_WEIGHTS=2.0,
            # Loss weights for surface parts. (24 Parts)
            PART_WEIGHTS=0.3,
            # Loss weights for UV regression.
            POINT_REGRESSION_WEIGHTS=0.5,
            openpose_train_weight=0,
            gt_train_weight=1
        ),   
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
            PRETRAINED_IM= '/workspace/dataset/3ddata/data/pretrained_model/resnet50-19c8e357.pth',
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
        type="BaseDataset",
        dataset="coco-full",
        batch_size=2,
        is_train=True
    ),
    train_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
    ),
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='AccuracyEval'), save_best='top_1')

# 导出配置
export=dict(
    input_shape_list = [[1,3,224,224]],
    input_name_list=["image"],
    output_name_list=["pred"]
)

max_epochs = 60