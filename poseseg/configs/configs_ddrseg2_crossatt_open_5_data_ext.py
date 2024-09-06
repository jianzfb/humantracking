# 优化器配置
base_lr = 0.01
optimizer = dict(
    type='AdamW', lr=base_lr,  weight_decay=0.05,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)
optimizer_config = dict(grad_clip=None)

max_epochs = 270
# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr*0.01,
    by_epoch=True,
    begin=100,
    end=max_epochs,
    warmup_iters=1000,
    warmup='linear',
    warmup_by_epoch = False
)

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

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
    type='PoseSegDDRNetV5',
    backbone=dict(
        type='BackboneDDR2',
        architecture='resnet34',
        in_channels=3,
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

num_keypoints = 33
# mapping
cocowholebody_betamp33 = [(0, 0), (1, 2), (2, 5), (3, 7), (4, 8), 
						  (5, 11), (6, 12), (7, 13), (8, 14), (9, 15), 
						  (10, 16), (11, 23), (12, 24), (13, 25), (14, 26), 
						  (15, 27), (16, 28), (17, 31), (19, 29), (20, 32),
						  (22, 30), (59, 6), (62, 4), (65, 1), (68, 3),
						  (71, 10), (77, 9), (93, 21), (96, 19), (108, 17), 
						  (114, 22), (117, 20), (129, 18)]

aic_betamp33 = [(0, 12), (1, 14), (2, 16), (3, 11), (4, 13), 
				(5, 15), (6, 24), (7, 26), (8, 28), (9, 23), (10, 25), (11, 27)]

crowdpose_betamp33 = [(0, 11), (1, 12), (2, 13), (3, 14), (4, 15), 
					  (5, 16), (6, 23), (7, 24), (8, 25), (9, 26), 
					  (10, 27), (11, 28)]

mpii_betamp33 = [(0, 28), (1, 26), (2, 24), (3, 23), (4, 25), 
				 (5, 27), (10, 16), (11, 14), (12, 12), (13, 11), 
				 (14, 13), (15, 15)]

jhmdb_betamp33 = [(3, 12), (4, 11), (5, 24), (6, 23), (7, 14), 
				  (8, 13), (9, 26), (10, 25), (11, 16), (12, 15), 
				  (13, 28), (14, 27)]

halpe_betamp33 = [(0, 0), (1, 2), (2, 5), (3, 7), (4, 8), 
				  (5, 11), (6, 12), (7, 13), (8, 14), (9, 15), 
				  (10, 16), (11, 23), (12, 24), (13, 25), (14, 26), 
				  (15, 27), (16, 28), (20, 31), (21, 32), (24, 29), 
				  (25, 30), (62, 6), (65, 4), (68, 1), (71, 3), 
				  (74, 10), (80, 9), (96, 21), (99, 19), (111, 17), 
				  (117, 22), (120, 20), (132, 18)]

ochuman_betamp33 = [(0, 0), (1, 2), (2, 5), (3, 7), (4, 8), 
					(5, 11), (6, 12), (7, 13), (8, 14), (9, 15), 
					(10, 16), (11, 23), (12, 24), (13, 25), (14, 26), 
					(15, 27), (16, 28)]

# 数据配置
data=dict(
    train=[
        [
            dict(
                type='CocoWholeBodyDataset',
                dir='/dataset/human_keypoints/coco-wholebody',
                ann_file='annotations/coco_wholebody_train_v1.0.json',
                data_prefix=dict(img='train2017/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=cocowholebody_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='HalpeDataset',
                dir='/dataset/human_keypoints/halpe',
                ann_file='annotations/halpe_train_v1.json',
                data_prefix=dict(img='hico_20160224_det/images/train2015/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=halpe_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='JhmdbDataset',
                dir='/dataset/human_keypoints/sub-JHMDB',
                ann_file='annotations/Sub1_train.json',
                data_prefix=dict(img=''),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=jhmdb_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='MpiiDataset',
                dir='/dataset/human_keypoints/mpii', 
                ann_file='annotations/mpii_trainval.json',
                data_prefix=dict(img='images/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=mpii_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='AicDataset',
                dir='/dataset/human_keypoints/aic', 
                ann_file='annotations/aic_train.json',
                data_prefix=dict(img='ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=aic_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='CrowdPoseDataset',
                dir='/dataset/human_keypoints/crowdpose',
                ann_file='annotations/mmpose_crowdpose_trainval.json',
                data_prefix=dict(img='images/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                )
            ),
            dict(
                type='OCHumanDataset',
                dir='/dataset/human_keypoints/OCHuman',
                ann_file='annotations/ochuman_coco_format_val_range_0.00_1.00.json',
                data_prefix=dict(img='images/'),
                pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=ochuman_betamp33),
                    dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                    dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.8,1.2,0.5], contrast=[0.8,1.2,0.5], brightness=[0.8,1.2,0.5]),
                    dict(
                        type='RandomFlipImage', 
                        swap_ids=[[
                            4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                        ],[
                            1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                        ]], 
                        swap_labels=[]
                    ),
                    dict(type='CoarseDropout', max_holes=1, max_height=0.3, max_width=0.3, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
                    dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                    dict(type="AddExtInfo"),
                    dict(type='ToTensor', keys=['image']),
                    dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
                ],
                inputs_def=dict(
                    fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
                ) 
            )
        ],
        dict(
            type='PersonTikTokAndAHPDataset',
            train_or_test='train',
            dir='/dataset',
            pipeline=[
                dict(type='ColorDistort', hue=[-5,5,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.5,1.3,0.5]),
                dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05),
                dict(type='ToTensor', keys=['image']),
                dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            ],
            inputs_def=dict(
                fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
            )
        ),
    ],
    train_dataloader=dict(
        samples_per_gpu=[64,64],        # 64,64
        workers_per_gpu=4,              # 4
        drop_last=True,
        shuffle=True,
    ),
    test=[
        dict(
            type='CrowdPoseDataset',
            dir='/workspace/dataset/human_keypoints/crowdpose',
            ann_file='annotations/mmpose_crowdpose_test.json',
            data_prefix=dict(img='images/'),
            pipeline=[
                dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_betamp33),
                dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
                dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05, with_random=False),
                dict(type="AddExtInfo"),
                dict(type='ToTensor', keys=['image']),
                dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            ],
            inputs_def=dict(
                fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'bboxes', 'has_segments'],
            )
        ),
        dict(
            type='PoseSeg_AHP',
            train_or_test='test',
            dir='/workspace/dataset/person-seg',
            pipeline=[
                dict(type="ConvertRandomObjJointsAndOffset", input_size=(256,192), heatmap_size=(64,48), num_joints=33, sigma=2, scale_factor=0.15,center_factor=0.05, with_random=False),
                dict(type='ToTensor', keys=['image']),
                dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            ],
            inputs_def=dict(
                fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'segments', 'has_joints', 'has_segments'],
            )
        ),
    ],
    test_dataloader=dict(
        samples_per_gpu=16, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
keypoint_weights=[
    1., 1., 1., 1., 1., 1., 1., 1.,1.,1.,1.,
    1.2, 1.2, 
    1.2, 1.2,
    1.2, 1.2, 
    1., 1., 
    1., 1.,
    1., 1.,
    1.1, 1.1,
    1.1, 1.1,
    1.5, 1.5,
    1.5, 1.5,
    1.5, 1.5
]
keypoint_sigmas=[
    0.026, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.035, 0.035, 0.035, 0.035, 
    0.079, 0.079,
    0.072, 0.072, 
    0.062, 0.062, 
    0.062, 0.062, 
    0.062, 0.062, 
    0.062, 0.062, 
    0.107, 0.107, 
    0.087, 0.087, 
    0.089, 0.089,
    0.089, 0.089,
    0.089, 0.089
]

evaluation = [
    # 评估关键点
    dict(out_dir='./output/', interval=1, metric=dict(type='OKS', sigmas=keypoint_sigmas, weights=keypoint_weights), save_best='oks', rule='greater'),
    # 评估分割
    dict(out_dir='./output/', interval=1, metric=dict(type='SegMIOU', class_num=2), save_best='miou', rule='greater')
]

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,192]],
    input_name_list=["image"],
    output_name_list=["heatmap", "offset", "seg"],
    deploy=dict(
        engine='rknn',      # rknn,snpe,tensorrt,tnn
        device='rk3588',    # rk3568/rk3588,qualcomm,nvidia,mobile
        preprocess=dict(
            mean_values=[0.491400*255, 0.482158*255, 0.4465231*255],        # mean values
            std_values=[0.247032*255, 0.243485*255, 0.2615877*255]          # std values
        ),
        quantize=False,                              # is quantize
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