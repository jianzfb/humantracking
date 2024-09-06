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
    begin=40,
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

# no EMA

# 模型配置
model = dict(
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

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=8)

# 数据配置
data=dict(
    train=dict(
        type='TFDataset',
        data_folder = [
            "/workspace/dataset/tuzhi/dataset",
            "/workspace/dataset/tuzhi/dataset",
            "/workspace/dataset/tuzhi/dataset",
            "/workspace/dataset/tuzhi/dataset",
            "/workspace/dataset/tuzhi/dataset",
        ],
        pipeline=[
            dict(type='DecodeImage', to_rgb=False),
            dict(type="ResizeS", target_dim=(1024,768)),    # 384,256
            dict(type="Rotation", degree=15),
            dict(type='RandomFlipImage', swap_labels=[]),
            dict(type="YoloBboxFormat"),
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='Permute', to_bgr=False, channel_first=True)
        ],
        description={'image': 'byte', 'bboxes': 'numpy', 'labels': 'numpy'},
        inputs_def=dict(
            fields = ["image", 'bboxes', 'image_meta']
        ),
        shuffle_queue_size=1024
    ),
    train_dataloader=dict(
        samples_per_gpu=4,        # 128
        workers_per_gpu=2,          # 4
        drop_last=True,
        shuffle=True,
        ignore_stack=['image', 'bboxes', 'image_meta']
    ),
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
    input_shape_list = [[1,3,768,1024]],
    input_name_list=["image"],
    output_name_list=["output"]
)