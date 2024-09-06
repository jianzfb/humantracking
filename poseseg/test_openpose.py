import enum
import sys
sys.path.insert(0,'/workspace/antgo')
from antgo.pipeline import *
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from enum import IntEnum
import math


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



skeleton = [
    [0,9],
    [0,10],
    [9,10],
    [11,12],
    [11,13],
    [12,14],
    [11,23],
    [12,24],
    [23,24],
    [23,25],
    [24,26],
    [25,27],
    [26,28],
    [27,29],
    [27,31],
    [31,29],
    [28,32],
    [28,30],
    [30,32],
    [13,15],
    [14,16]
]

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [
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
    ]


def draw_bodypose(index, image, all_person_poses):
    h,w = image.shape[:2]

    for person_i in range(len(all_person_poses)):
        person_pose = all_person_poses[person_i]
        for point_i in range(33):
            x,y,score = person_pose[point_i]
            if score > 0.01:
                x = x/96.0 * w
                y = y/64.0 * h

                if point_i != 0:
                    image = cv2.circle(image, (int(x), int(y)), 1, (0,0,255),1)
                else:
                    image = cv2.circle(image, (int(x), int(y)), 2, (0,255,0),2)

        for s,e in skeleton:
            start_x,start_y, start_s = person_pose[s]
            end_x, end_y, end_s = person_pose[e]
            if start_s < 0.01 or end_s < 0.01:
                continue

            cv2.line(image, (int(start_x/96*w), int(start_y/64*h)), (int(end_x/96*w), int(end_y/64*h)), (255,0,0), 1)

    cv2.imwrite(f"./temp/{index}.png", image)
    return image


def decode_pose(image, paf_avg, heatmap_avg):
    all_peaks = []
    peak_counter = 0
    thre1 = 0.02
    thre2 = 0.015

    for part in range(33):
        map_ori = heatmap_avg[0, part, :, :]
        # one_heatmap = gaussian_filter(map_ori, sigma=3)
        one_heatmap = map_ori

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse

        # 基于加权计算浮点坐标
        refine_peaks = []
        for i_x,i_y in peaks:
            f_x = 0
            f_y = 0
            f_c = 0
            for ii in range(i_y-3,i_y+3):
                for jj in range(i_x-3,i_x+3):
                    if ii >= 0 and ii < 64 and jj >= 0 and jj < 96:
                        f_c += one_heatmap[ii,jj]
                        f_x += jj * one_heatmap[ii,jj]
                        f_y += ii * one_heatmap[ii,jj]

            f_x = f_x / f_c
            f_y = f_y / f_c
            if f_x > 95:
                f_x = 95
            if f_y > 63:
                f_y = 63
            if f_x < 0:
                f_x = 0
            if f_y < 0:
                f_y = 0
            refine_peaks.append((f_x,f_y))
            # refine_peaks.append((i_x,i_y))

        peaks_with_score = [refine_x + (map_ori[x[1], x[0]],) for x, refine_x in zip(peaks,refine_peaks)]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(limbSeq)):
        score_mid = paf_avg[0, k*2:(k+1)*2, :, :]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[0, int(round(startend[I][1])), int(round(startend[I][0]))] \
                                    for I in range(len(startend))])
                    vec_y = np.array([score_mid[1, int(round(startend[I][1])), int(round(startend[I][0]))] \
                                    for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.5 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[3], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # 拆分出人
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    person_num = len(all_peaks[0])
    connection_by_group = [[] for _ in range(person_num)]

    all_person_poses = []
    for person_i in range(person_num):
        start_peak_i = all_peaks[0][person_i][3]
        nose_peak_i = start_peak_i

        # face region
        # nose, mouth, eye
        check_limb_index_list = [0,1,2,3]
        left_eye_peak_i = -1
        right_eye_peak_i = -1
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == nose_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    if k == 2:
                        left_eye_peak_i = b_peak_i
                    if k == 3:
                        right_eye_peak_i = b_peak_i

        check_limb_index_list = [4,5,6]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == left_eye_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))

        check_limb_index_list = [7,8,9]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == right_eye_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))

        # left part:
        # left: nose->shoulder->...->foot
        root_peak_i = nose_peak_i
        shoulder_peak_i = -1
        check_limb_index_list = [10, 12, 13, 14]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    root_peak_i = b_peak_i
                    if k == 10:
                        shoulder_peak_i = b_peak_i
                    break

        # left: foot->foot tip
        # left: foot->foot hill
        check_limb_index_list = [15,16]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue

            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    break

        # left: shoulder->elbow->hand
        root_peak_i = shoulder_peak_i
        hand_peak_i = -1
        check_limb_index_list = [17,18]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    root_peak_i = b_peak_i
                    if k == 18:
                        hand_peak_i = b_peak_i
                    break

        # left: hand->plam1
        # left: hand->plam2
        # left: hand->plam3
        root_peak_i = hand_peak_i
        check_limb_index_list = [19,20,21]        
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    break

        # right part
        # right: nose->shoulder->...->foot
        root_peak_i = nose_peak_i
        shoulder_peak_i = -1
        check_limb_index_list = [11, 22, 23, 24]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    root_peak_i = b_peak_i
                    if k == 11:
                        shoulder_peak_i = b_peak_i
                    break

        # right: foot->foot tip
        # right: foot->foot hill
        check_limb_index_list = [25,26]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    break

        # right: shoulder->elbow->hand
        root_peak_i = shoulder_peak_i
        hand_peak_i = -1
        check_limb_index_list = [27,28]
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    root_peak_i = b_peak_i
                    if k == 28:
                        hand_peak_i = b_peak_i
                    break

        # right: hand->plam1
        # right: hand->plam2
        # right: hand->plam3
        root_peak_i = hand_peak_i
        check_limb_index_list = [29,30,31]        
        for k in check_limb_index_list:
            connection = connection_all[k]
            if len(connection) == 0:
                continue
            for a_peak_i, b_peak_i in connection[:, :2]:
                if a_peak_i == root_peak_i:
                    connection_by_group[person_i].append((k, a_peak_i, b_peak_i))
                    break
                    
        person_pose = np.zeros((33,3))
        for person_i_connections in connection_by_group[person_i]:
            k, a_peak_i, b_peak_i = person_i_connections
            a_x,a_y,a_score = candidate[int(a_peak_i)][:3]
            b_x,b_y,b_score = candidate[int(b_peak_i)][:3]

            person_pose[limbSeq[k][0]] = [a_x,a_y,a_score]
            person_pose[limbSeq[k][1]] = [b_x,b_y,b_score]

        all_person_poses.append(person_pose)
    return all_person_poses,

def debug_show(image, pose):
    image_h, image_w = image.shape[:2]
    # heatmap_h, heatmap_w = heatmap.shape[2:]

    # pose_points = []
    # num_joints = 33
    # for channel_i in range(num_joints):
    #     aa = np.argmax(heatmap[0,channel_i])
    #     r = aa // heatmap_w
    #     c = aa % heatmap_w
    #     if heatmap[0,channel_i,r,c] < 0.2:
    #         pose_points.append((-1,-1))
    #         continue

    #     # offset_x = offset[0, channel_i, r, c]
    #     # offset_y = offset[0, num_joints+channel_i, r, c]

    #     # point_x = c + offset_x
    #     # point_y = r + offset_y
    #     point_x = c
    #     point_y = r

    #     point_x = point_x * (image_w/heatmap_w)
    #     point_y = point_y * (image_h/heatmap_h)
    #     image = cv2.circle(image, (int(point_x), int(point_y)), 1, (0,0,255),1)
    #     pose_points.append((point_x, point_y))
    
    # for s,e in skeleton:
    #     start_x,start_y = pose_points[s]
    #     end_x, end_y = pose_points[e]
    #     if start_x < 0 or end_x < 0:
    #         continue

    #     cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,0,0), 1)
    # cv2.imwrite('./aabb.png', image)
    # return image
    return None


# image = cv2.imread('/workspace/humantracking/11.png')

video_dc['image', 'index']('/workspace/humantracking/2024_04_03_16_58_10.mp4'). \
    resize_op['image', 'resized_image_for_pose'](out_size=(384,256)). \
    inference_onnx_op['resized_image_for_pose', ("paf", "heatmap")](
        onnx_path='/workspace/humantracking/poseseg-epoch_100_2-model.onnx', 
        mean=[0.491400*255, 0.482158*255, 0.4465231*255],
        std=[0.247032*255, 0.243485*255, 0.2615877*255],
        engine='rknn',
        engine_args={
            'device':'rk3588',
        }
    ). \
    runas_op[('image', "paf", "heatmap"), ('all_person_poses',)](func=decode_pose). \
    runas_op[('index','image', 'all_person_poses'), 'out'](func=draw_bodypose).run()


# # 测试rknn结果
# # 创建rknn C++ 工程
# placeholder['image'](image). \
#     inference_onnx_op['image', ("cls_scores", "bbox_preds", "kpt_vis", "pose_vecs")](
#         onnx_path='/workspace/humantracking/poseseg-epoch_280-model.onnx', 
#         mean=[0.491400*255, 0.482158*255, 0.4465231*255],
#         std=[0.247032*255, 0.243485*255, 0.2615877*255],
#         engine='rknn',
#         engine_args={
#             'device':'rk3588',
#             'quantize': False
#         }). \
#     build(
#         platform='android/arm64-v8a',
#         project_config={
#             'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
#             'output': [
#                 ('cls_scores', 'EAGLEEYE_SIGNAL_TENSOR'),
#                 ('bbox_preds', 'EAGLEEYE_SIGNAL_TENSOR'),
#                 ('kpt_vis', 'EAGLEEYE_SIGNAL_TENSOR'),
#                 ('pose_vecs', 'EAGLEEYE_SIGNAL_TENSOR')
#             ],
#             'name': 'temptest',
#             'git': ''
#         }
#     )

# 测试工程
# placeholder['image'](image). \
#     eagleeye.exe.temptest['image', ('cls_scores', 'bbox_preds', 'kpt_vis', 'pose_vecs')](). \
#     runas_op[('image', "cls_scores", "bbox_preds", "kpt_vis", "pose_vecs"), 'out'](func=debug_show).run()


print('hello')

