import numpy as np

# 优化器配置
base_lr = 0.005
optimizer = dict(
    type='AdamW', lr=base_lr,  weight_decay=0.05,
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0, 
        bypass_duplicate=True
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

keypoint_weights=[
    6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
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

max_epochs = 130
# 学习率调度配置
lr_config = [
    dict(
        policy='Warmup',
        begin=0,
        end=5,
        warmup_iters=5,
        warmup='linear',
        warmup_by_epoch=False
    ),
    dict(
        policy='CosineAnnealing',
        min_lr=0.0005,
        by_epoch=True,
        begin=5,
        end=120,
    ),
    dict(policy='Fixed', by_epoch=True, factor=1, begin=120, end=130),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256) # 256

# 自定义HOOKS
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

# 日志配置
log_config = dict(
    interval=10,         # 10
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)


widen_factor = 1.0
deepen_factor = 1.0

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='relu'),
    backbone=dict(
        type='SKetNetF',
        architecture='resnet34',
        in_channels=3,
        out_indices=[2,3,4]
    ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[96, 128, 160],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=128,
        output_indices=[1, 2],
        num_encoder_layers=0,
        use_encoder_idx=[],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='ReLU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[128, 128],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2),
        act_cfg=dict(type='ReLU', inplace=True)
    ),
    head=dict(
        type='RTMOHead',
        num_keypoints=33,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=128,
            cls_feat_channels=128,
            channels_per_group=36,
            pose_vec_channels=-1,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='ReLU')),
        assigner=dict(
            type='SimOTAAssigner2',
            cls_weight=0.0,
            oks_weight=0.0,
            iou_weight=3.0,
            dynamic_k_indicator='iou',
            oks_calculator=dict(type='PoseOKS', sigmas=keypoint_sigmas)),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=8.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='mean',
            norm_target_weight=True,
            sigmas=keypoint_sigmas,
            loss_weight=40.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='none',
            loss_weight=1.0),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
        keypoint_weights=np.array(keypoint_weights, dtype=np.float32)
    ),
    test_cfg=dict(
        input_size=(384,384),
        score_thr=0.3,
        nms_thr=0.4,
    )
)


# checkpoint配置
checkpoint_config = dict(interval=5, out_dir='./output/')       


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


eval_crowdpose_betamp33 = [
    (11,11),(12,12),(13,13),(14,14),(15,15),
    (16,16),(23,23),(24,24),(25,25),(26,26),
    (27,27),(28,28)
]

# 数据配置
data=dict(
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoWholeBodyDataset',
                dir='/dataset/human_keypoints/coco-wholebody',
                ann_file='annotations/coco_wholebody_train_v1.0.json',
                data_prefix=dict(img='train2017/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=cocowholebody_betamp33),
                ]
            ),
            dict(
                type='HalpeDataset',
                dir='/dataset/human_keypoints/halpe',
                ann_file='annotations/halpe_train_v1.json',
                data_prefix=dict(img='hico_20160224_det/images/train2015/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=halpe_betamp33),
                ]
            ),
            dict(
                type='JhmdbDataset',
                dir='/dataset/human_keypoints/sub-JHMDB',
                ann_file='annotations/Sub1_train.json',
                data_prefix=dict(img=''),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=jhmdb_betamp33),
                ]
            ),
            dict(
                type='MpiiDataset',
                dir='/dataset/human_keypoints/mpii', 
                ann_file='annotations/mpii_trainval.json',
                data_prefix=dict(img='images/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=mpii_betamp33),
                ]
            ),
            dict(
                type='AicDataset',
                dir='/dataset/human_keypoints/aic', 
                ann_file='annotations/aic_train.json',
                data_prefix=dict(img='ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=aic_betamp33)
                ]
            ),
            dict(
                type='CrowdPoseDataset',
                dir='/dataset/human_keypoints/crowdpose',
                ann_file='annotations/mmpose_crowdpose_trainval.json',
                data_prefix=dict(img='images/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_betamp33),
                ]
            ),
            dict(
                type='OCHumanDataset',
                dir='/dataset/human_keypoints/OCHuman',
                ann_file='annotations/ochuman_coco_format_val_range_0.00_1.00.json',
                data_prefix=dict(img='images/'),
                data_mode='bottomup',
                pre_pipeline=[
                    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=ochuman_betamp33),
                ]
            ),      
        ],
        # datasets=[
        #     dict(
        #         type='CrowdPoseDataset',
        #         dir='/workspace/dataset/human_keypoints/crowdpose',
        #         ann_file='annotations/mmpose_crowdpose_trainval.json',
        #         data_prefix=dict(img='images/'),
        #         data_mode='bottomup',
        #         pre_pipeline=[
        #             dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_betamp33),
        #         ],
        #     )
        # ],
        pipeline=[
            dict(
                type='Mosaic',
                img_scale=(384, 384),
                pad_val=0
            ),
            dict(
                type='BottomupRandomAffine',
                input_size=(384, 384),
                shift_factor=0.1,
                rotate_factor=15,
                scale_factor=(1.0,1.5),
                pad_val=0,
                distribution='uniform',
                transform_mode='affine_udp',
                bbox_keep_corner=False,
                clip_border=True,
            ),
            dict(type='ColorDistort', hue=[-15,15,0.5], saturation=[0.5,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.7,1.3,0.5]),
            dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False, min_kpt_vis=4, min_gt_bbox_wh=(10,20)),
            dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis'}),
            dict(
                type='RandomFlipImage',
                swap_ids=[[
                    4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32
                ],[
                    1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31
                ]], 
                swap_labels=[]
            ),
            dict(type='KeynameConvert', mapping={'bboxes': 'bbox', 'joints2d': 'keypoints', 'joints_vis':'keypoints_visible'}),
            dict(type='GenerateTarget', encoder=dict(input_size=(384,384))),
            dict(type='KeynameConvert', mapping={'bbox': 'bboxes', 'keypoints': 'joints2d', 'keypoints_visible': 'joints_vis', 'keypoints_visible_weights': 'joints_weight'}),
            dict(type='CorrectBoxes'),
            dict(type='CoarseDropout', max_holes=1, max_height=0.2, max_width=0.2, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'bboxes', 'joints_vis', 'joints_weight', 'joints2d', 'bbox_labels', 'area'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=32, # 32
        workers_per_gpu=4,  # 4
        drop_last=True,
        shuffle=True,
        ignore_stack=['bboxes', 'joints_vis', 'joints_weight', 'joints2d', 'bbox_labels', 'area']
    ),
    test=dict(
        type='CrowdPoseDataset',
        dir='/workspace/dataset/human_keypoints/crowdpose',
        ann_file='annotations/mmpose_crowdpose_test.json',
        data_prefix=dict(img='images/'),
        data_mode='bottomup',
        pre_pipeline=[
            dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=crowdpose_betamp33),
        ],
        pipeline=[
            dict(type='BottomupResize', input_size=(384,384), pad_val=(0, 0, 0), bbox_keep_corner=False),             
            dict(type='GenerateTarget', encoder=dict(input_size=(384,384))),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'bbox', 'keypoints_visible', 'keypoints', 'bbox_labels', 'area', 'raw_ann_info', 'img_id', 'image_meta', 'inv_warp_mat'],
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
        ignore_stack=['bbox', 'keypoints_visible', 'keypoints', 'bbox_labels', 'area', 'raw_ann_info', 'img_id', 'image_meta', 'inv_warp_mat']
    )
)

# 评估方案配置
evaluation=dict(
    out_dir='./output/', 
    interval=1, 
    metric=dict(
        type='CocoMetric',  
        num_keypoints=33, 
        keypoint_sigmas=np.array(keypoint_sigmas),
        gt_converter=dict(
            num_keypoints=33,
            mapping=crowdpose_betamp33
        ),
        pred_converter=dict(
            num_keypoints=33,
            mapping=eval_crowdpose_betamp33
        )
    ), 
    rule='greater'
)


# 导出配置
export=dict(
    input_shape_list = [[1,3,384,384]],
    input_name_list=["image"],
    output_name_list=["cls_scores", "bbox_preds", "kpt_vis", "kpt_offset"],
)

seed=5