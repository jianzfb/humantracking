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
    type="TransPose",
    train_cfg=dict()
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