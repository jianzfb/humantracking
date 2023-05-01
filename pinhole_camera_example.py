import os

import cv2
import numpy as np
from humantracking.camera_model import OmniCamera
from humantracking.perspective_utils import perspective_crop_image, resize_preserve_aspect_ratio
from tqdm import tqdm

HAND_NUM = 2
HAND_COLOR = [(255, 0, 0), (0, 255, 0)]
CAMERA_ROTATE = {
    # 'upper_left_view': cv2.ROTATE_90_CLOCKWISE,
    #  'upper_right_view': cv2.ROTATE_90_COUNTERCLOCKWISE,
    "right_view": cv2.ROTATE_90_CLOCKWISE,
    #  'left_view': cv2.ROTATE_90_COUNTERCLOCKWISE
}


def load_onmi_camera_from_list(camera_parameter_list):

    lines_per_camera = 25

    camera_params = list(map(float, camera_parameter_list))

    camera_dict = {}
    for i, camera_id in enumerate(CAMERA_ROTATE):
        camera = OmniCamera(camera_params[i * lines_per_camera : (i + 1) * lines_per_camera])
        camera_dict[camera_id] = camera

    return camera_dict


def load_onmi_camera_from_file(camera_parameter_path):
    with open(camera_parameter_path, "r") as txt_file:
        camera_params = txt_file.readlines()

    return load_onmi_camera_from_list(camera_params)


if __name__ == "__main__":
    data_root = (
        "/mnt/bn/wlxlmk/mlx/users/huhui.22/playground/aipack/aipack/products/ai/mvdetect3d/perspective_crop/data"
    )
    img_dir = os.path.join(data_root, "input")
    label_dir = os.path.join(data_root, "output")

    camera_parameter_path = os.path.join(data_root, "params.txt")
    camera_dict = load_onmi_camera_from_file(camera_parameter_path)

    crop_size = (128, 128)

    start_idx = 41
    load_num = 4
    for img_idx in tqdm(range(start_idx, start_idx + load_num)):
        img_vis_list = []
        for camera_name in CAMERA_ROTATE:
            img_name = "inimg_{}_{}.png".format(str(img_idx).zfill(8), camera_name)
            png_path = os.path.join(img_dir, img_name)
            img = cv2.imread(png_path)
            img_vis = img.copy()

            # boxes = boxes[boxes[:, 0] != -1]
            crop_list = []
            for hand_id in range(1):
                label_name = "box_{}_{}_{}.txt".format(
                    str(img_idx).zfill(8), camera_name, ["left_hand", "right_hand"][hand_id]
                )
                label_path = os.path.join(label_dir, label_name)
                hand_boxes = np.loadtxt(label_path).reshape(1, -1)
                hand_boxes[:, 2] = hand_boxes[:, 0] + hand_boxes[:, 2]
                hand_boxes[:, 3] = hand_boxes[:, 1] + hand_boxes[:, 3]
                # hand_boxes = boxes[boxes[:, 4] == hand_id]
                # hand_boxes = hand_boxes[hand_boxes[:, 5].argsort()[::-1]]
                if hand_boxes.shape[0] == 0:
                    continue

                bb = hand_boxes[0, :4].astype(np.int32)
                cv2.rectangle(img_vis, pt1=(bb[0], bb[1]), pt2=(bb[2], bb[3]), color=HAND_COLOR[hand_id], thickness=5)
                affine_crop = img[bb[1] : bb[3] + 1, bb[0] : bb[2] + 1]
                # affine_crop = cv2.resize(affine_crop, crop_size)
                affine_crop = resize_preserve_aspect_ratio(affine_crop, crop_size)
                # affine_crop = cv2.rotate(affine_crop, CAMERA_ROTATE[camera_name])
                crop_list.append(affine_crop)

                perspect_crop, perspect_points = perspective_crop_image(
                    img,
                    bb,
                    crop_size,
                    camera_model=camera_dict[camera_name],
                )
                for p in perspect_points:
                    if p[0] < 0 or p[1] < 0:
                        continue
                    cv2.circle(img_vis, (p[0], p[1]), color=HAND_COLOR[hand_id], radius=3, thickness=5)
                # perspect_crop = cv2.rotate(perspect_crop, CAMERA_ROTATE[camera_name])
                crop_list.append(perspect_crop)

            # img_vis = cv2.rotate(img_vis, CAMERA_ROTATE[camera_name])
            empty_column = np.zeros((crop_size[0] * 4, crop_size[1], 3)).astype(np.uint8)
            for cdx, crop in enumerate(crop_list):
                empty_column[crop_size[0] * cdx : crop_size[0] * (cdx + 1), : crop_size[1]] = crop
            column_width = int(img.shape[0] * empty_column.shape[1] / empty_column.shape[0])
            empty_column = cv2.resize(empty_column, (column_width, img_vis.shape[0]))
            img_vis = np.concatenate([img_vis, empty_column], axis=1)
            img_vis_list.append(img_vis)
        # vis_row1 = np.concatenate(img_vis_list[:2], axis=1)
        # vis_row2 = np.concatenate(img_vis_list[2:], axis=1)
        # whole_vis = np.concatenate([vis_row1, vis_row2], axis=0)
        # whole_vis = cv2.resize(whole_vis, (whole_vis.shape[1]//2, whole_vis.shape[0]//2))
        # cv2.imshow(str(img_idx), whole_vis)
        cv2.imwrite("outputs/" + str(img_idx).zfill(5) + ".png", img_vis_list[0])
        # cv2.waitKey(0)
        # from IPython import embed; embed(); exit(); # BREAKPOINT
