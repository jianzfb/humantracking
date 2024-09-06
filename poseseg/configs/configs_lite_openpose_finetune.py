from enum import IntEnum
import numpy as np

class JointType(IntEnum):
    """ 鼻 """
    Nose = 0
    """ 左嘴角 """
    LeftMouth = 9
    """ 右嘴角 """
    RightMouth = 10
    """ 左目 """
    LeftEye = 2
    """ 右目 """
    RightEye = 5
    """ 左内眼角 """
    LeftInnerEyeCorner = 1
    """ 左外眼角 """
    LeftOuterEyeCorner = 3
    """ 右内眼角 """
    RightInnerEyeCorner = 4
    """ 右外眼角 """
    RightOuterEyeCorner = 6
    """ 左耳 """
    LeftEar = 7
    """ 右耳 """
    RightEar = 8
    """ 左肩 """
    LeftShoulder = 11
    """ 右肩 """
    RightShoulder = 12
    """ 左肘 """
    LeftElbow = 13
    """ 右肘 """
    RightElbow = 14
    """ 左手 """
    LeftHand = 15
    """ 右手 """
    RightHand = 16
    """ 左手 palm 1 """
    LeftPlam1 = 17
    """ 左手 palm 2 """
    LeftPlam2 = 19
    """ 左手 palm 3 """
    LeftPlam3 = 21
    """ 右手 palm 1 """
    RightPlam1 = 18
    """ 右手 palm 2 """
    RightPlam2 = 20
    """ 右手 palm 3 """
    RightPlam3 = 22

    """ 左腰 """
    LeftWaist = 23
    """ 右腰 """
    RightWaist = 24
    """ 左膝 """
    LeftKnee = 25
    """ 右膝 """
    RightKnee = 26
    """ 左足 """
    LeftFoot = 27
    """ 右足 """
    RightFoot = 28
    """ 左足脚尖 """
    LeftFootTip = 31
    """ 右足脚尖 """
    RightFootTip = 32
    """ 左足脚跟 """
    LeftFootHill = 29
    """ 右足脚跟 """
    RightFootHill = 30


params = {
    'paf_sigma': 2,
    'heatmap_sigma': 1.5,
    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,
    'limbs_point': [
        [JointType.Nose, JointType.LeftMouth],
        [JointType.Nose, JointType.RightMouth],
        [JointType.Nose, JointType.LeftEye],
        [JointType.Nose, JointType.RightEye],
        [JointType.LeftEye, JointType.LeftInnerEyeCorner],
        [JointType.LeftEye, JointType.LeftOuterEyeCorner],
        [JointType.LeftEye, JointType.LeftEar],
        [JointType.RightEye, JointType.RightInnerEyeCorner],
        [JointType.RightEye, JointType.RightOuterEyeCorner],
        [JointType.RightEye, JointType.RightEar],

        [JointType.Nose, JointType.LeftShoulder],
        [JointType.Nose, JointType.RightShoulder],

        [JointType.LeftShoulder, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.LeftFoot, JointType.LeftFootTip],
        [JointType.LeftFoot, JointType.LeftFootHill],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftHand, JointType.LeftPlam1],
        [JointType.LeftHand, JointType.LeftPlam2],
        [JointType.LeftHand, JointType.LeftPlam3],

        [JointType.RightShoulder, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.RightFoot, JointType.RightFootTip],
        [JointType.RightFoot, JointType.RightFootHill],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightHand, JointType.RightPlam1],
        [JointType.RightHand, JointType.RightPlam2],
        [JointType.RightHand, JointType.RightPlam3],
    ],
}

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
optimizer_config = dict(grad_clip=dict(max_norm=0.01))

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

max_epochs = 120
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

joint_weight = np.ones((1,33), dtype=np.float32)
joint_weight[0,:11] = 1.5
limb_weight = np.ones((1,64), dtype=np.float32)
limb_weight[0, :12*2] = 1.5
model = dict(
    type='PoseDDROpenPose',
    backbone=dict(
        type='BackboneDeltaDDRM',
        in_channels=3
    ),
    train_cfg=dict(
        joint_weight=joint_weight,
        limb_weight=limb_weight
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
            )
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
                input_size=(384, 256),
                shift_factor=0.1,
                rotate_factor=15,
                scale_factor=(1.0,1.5),
                pad_val=0,
                distribution='uniform',
                transform_mode='affine_udp',
                bbox_keep_corner=False,
                clip_border=True,
            ),
            dict(type='ColorDistort', hue=[-18,18,0.5], saturation=[0.7,1.3,0.5], contrast=[0.7,1.3,0.5], brightness=[0.7,1.3,0.5]),
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
            dict(type='CoarseDropout', max_holes=1, max_height=0.2, max_width=0.2, min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
            dict(type='WithPAFGenerator', params=params, insize=(384,256), stride=4),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'pafs', 'heatmaps', 'ignore_mask', 'heatmap_valid_mask', 'paf_valid_mask'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=32, # 32
        workers_per_gpu=4,  # 4
        drop_last=True,
        shuffle=True,
    ),
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='OKS', sigmas=keypoint_sigmas, weights=keypoint_weights), save_best='oks', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,384]],
    input_name_list=["image"],
    output_name_list=['paf', "heatmap"],
)

seed=5