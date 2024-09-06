from antgo.dataflow.imgaug import *
from antgo.framework.helper.dataset.builder import PIPELINES
import numpy as np
import cv2
from models.codecs.simcc_label import SimCCLabel
import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from scipy.stats import truncnorm


@PIPELINES.register_module()
class SimCCLabelEncoder(object):
    def __init__(self, input_size=(256,256), sigma=(6,6.93),simcc_split_ratio=2.0,normalize=False, use_dark=False, inputs=None):
        self.sim_cc_label = SimCCLabel(
            input_size=input_size,
            sigma=sigma,
            simcc_split_ratio=simcc_split_ratio,
            normalize=normalize,
            use_dark=use_dark
        )

    def __call__(self, sample):
        keypoints = sample['joints2d']
        keypoints_visible = sample['joints_vis']
        cc_encoded = self.sim_cc_label.encode(keypoints, keypoints_visible)

        return {
            'image': sample['image'],
            'keypoint_x_labels': cc_encoded['keypoint_x_labels'],
            'keypoint_y_labels': cc_encoded['keypoint_y_labels'],
            'keypoint_weights': cc_encoded['keypoint_weights']
        }

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


@PIPELINES.register_module()
class RandomBBoxTransform(object):
    def __init__(self, input_size=(192,256), 
                 shift_factor=0.16,
                 shift_prob=0.3,
                 scale_factor=(0.5, 1.5),
                 scale_prob=1.0,
                 rotate_factor=80.0,
                 rotate_prob=0.6):
        self.input_size = input_size
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        random_v = self._truncnorm(size=(num_bboxes, 4))
        offset_v = random_v[:, :2]
        scale_v = random_v[:, 2:3]
        rotate_v = random_v[:, 3]

        # Get shift parameters
        offset = offset_v * self.shift_factor
        offset = np.where(
            np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = scale_v * sigma + mu
        scale = np.where(
            np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        rotate = rotate_v * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale
    def get_warp_matrix(self,
                        center: np.ndarray,
                        scale: np.ndarray,
                        rot: float,
                        output_size: Tuple[int, int],
                        shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        """Calculate the affine transformation matrix that can warp the bbox area
        in the input image to the output size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: A 2x3 transformation matrix
        """
        assert len(center) == 2
        assert len(scale) == 2
        assert len(output_size) == 2
        assert len(shift) == 2

        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)
        src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat


    def __call__(self, sample):
        num_bboxes = sample['bboxes'].shape[0]
        bbox_center = (sample['bboxes'][..., 2:] + sample['bboxes'][..., :2]) * 0.5
        bbox_scale = (sample['bboxes'][..., 2:] - sample['bboxes'][..., :2]) * 1.25
        offset, scale, rotate = self._get_transform_params(num_bboxes)
        bbox_center = bbox_center + offset * bbox_scale
        bbox_scale = bbox_scale * scale
        bbox_rotation = rotate

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        bbox_scale = self._fix_aspect_ratio(
            bbox_scale, aspect_ratio=w / h)

        center = bbox_center[0]
        scale = bbox_scale[0]
        rot = bbox_rotation[0]
        warp_mat = self.get_warp_matrix(center, scale, rot, output_size=(w, h))
        sample['image'] = cv2.warpAffine(
                sample['image'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)
        sample['joints2d'] = cv2.transform(
                sample['joints2d'][..., :2], warp_mat)
        
        joints_vis = sample['joints_vis']
        check_mask = (sample['joints2d'][...,0] >= 0) * (sample['joints2d'][..., 0] < w) * (sample['joints2d'][...,1] >= 0) * (sample['joints2d'][..., 1] < h)
        joints_vis[np.where(check_mask == False)] = 0        
        sample['joints_vis'] = joints_vis

        # image = sample['image']
        # for joint_i, (x,y) in enumerate(sample['joints2d'][0]):
        #     x, y = int(x), int(y)
        #     if joints_vis[0,joint_i]:
        #         cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)
        #         cv2.putText(image, f'{joint_i}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        # cv2.imwrite(f'./check.png', image)

        return sample


@PIPELINES.register_module()
class RandomAugForPose(object):
    def __init__(self, input_size, heatmap_size, num_joints, sigma=2, scale_factor=0.5, center_factor=0.25, rot_factor=30, skeleton=[], with_random=True, inputs=None):
        self.input_size = input_size
        self._heatmap_size = heatmap_size
        self._scale_factor = scale_factor
        self._center_factor = center_factor
        self.with_random = with_random 
        self._rot_factor = rot_factor
        self.num_joints = num_joints
        self._feat_stride = np.array(self.input_size) / np.array(self._heatmap_size) 
        self._sigma = sigma
        self.skeleton = skeleton

    def extend_bbox(self, bbox, image_shape=None):
        x1, y1, x2, y2 = bbox
        wb, hb = x2 - x1, y2 - y1
        scale = 0.15

        if image_shape is not None:
            height, width = image_shape
            newx1 = x1 - wb * scale if x1 - wb * scale > 0 else 0
            newx2 = x2 + wb * scale if x2 + wb * scale < width else width - 1
            newy1 = y1 - hb * scale if y1 - hb * scale > 0 else 0
            newy2 = y2 + hb * scale if y2 + hb * scale < height else height - 1
        else:
            newx1 = x1 - wb * scale
            newx2 = x2 + wb * scale
            newy1 = y1 - hb * scale
            newy2 = y2 + hb * scale
        exbox = [int(newx1), int(newy1), int(newx2), int(newy2)]
        return exbox
  
    def _cal_offset(self, heatmap, roi, pt_ori):
        offset_x = np.zeros((heatmap.shape[0], heatmap.shape[1]))
        offset_y = np.zeros((heatmap.shape[0], heatmap.shape[1]))
        weight = np.zeros((heatmap.shape[0], heatmap.shape[1]))
        weight[heatmap != 0] = 1

        for x in range(roi[0], roi[2]):
            offset_x[roi[1] : roi[3], x] = pt_ori[0] - x
        for y in range(roi[1], roi[3]):
            offset_y[y, roi[0] : roi[2]] = pt_ori[1] - y

        return offset_x, offset_y, weight

    def _box_to_center_scale(self, x, y, w, h, aspect_ratio=1.0, scale_mult=1.0):
        """Convert box coordinates to center and scale.
        adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
        """
        pixel_std = 1
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * scale_mult
        return center, scale

    def get_dir(self, src_point, rot_rad):
        """Rotate the point by `rot_rad` degree."""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        """Return vector c that perpendicular to (a - b)."""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_affine_transform(self, center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.0]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def __call__(self, sample, context=None):
        image = sample['image']
        joints2d = sample['joints2d']
        joints_vis = sample['joints_vis']
        bbox = np.zeros((4), dtype=np.int32)
        if len(joints2d.shape) == 3:
            obj_num = joints2d.shape[0]
            obj_i = np.random.randint(0, obj_num)
            joints2d = joints2d[obj_i]
            joints_vis = joints_vis[obj_i]

            if 'bboxes' in sample and len(sample['bboxes']) > 0:
                bbox = sample['bboxes'][obj_i]
            else:
                x1, y1, x2, y2 = [joints2d[:, 0].min(), joints2d[:, 1].min(), joints2d[:, 0].max(), joints2d[:, 1].max()]
                bbox = [x1, y1, x2, y2]
        else:
            if 'bboxes' in sample and len(sample['bboxes']) > 0:
                bbox = sample['bboxes']
            else:
                x1, y1, x2, y2 = [joints2d[:, 0].min(), joints2d[:, 1].min(), joints2d[:, 0].max(), joints2d[:, 1].max()]
                bbox = [x1, y1, x2, y2]

        # 动态扩展bbox
        bbox = self.extend_bbox(bbox, image.shape[:2])
        xmin, ymin, xmax, ymax = bbox
        center, scale = self._box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin, 1.0)
        if self.with_random:
            sf = self._scale_factor
            ran_tmp = np.clip((np.random.rand() - 0.5) * 2.0 * sf + 1.0, 1 - sf, 1 + sf)
            scale = scale * ran_tmp

        if self.with_random:
            cf = self._center_factor
            dx = (xmax - xmin) * np.random.uniform(-cf, cf)
            dy = (ymax - ymin) * np.random.uniform(-cf, cf)
            center[0] += dx
            center[1] += dy

        r = 0.0
        if self.with_random:
            rf = self._rot_factor
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if np.random.uniform(0, 1) <= 0.7 else 0

        inp_h, inp_w = self.input_size
        trans = self.get_affine_transform(center, scale, r, [inp_w, inp_h])

        # 裁减图像
        image = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        if 'segments' in sample:
            sample['segments'] = cv2.warpAffine(sample['segments'], trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # 转换2D关键点
        for i in range(self.num_joints):
            joints2d[i, 0:2] = self.affine_transform(joints2d[i, 0:2], trans)

        check_mask = (joints2d[:,0] >= 0) * (joints2d[:, 0] < inp_w) * (joints2d[:,1] >= 0) * (joints2d[:, 1] < inp_h)
        joints_vis[np.where(check_mask == False)] = 0

        # 修正bbox
        x1, y1, x2, y2 = [joints2d[:, 0].min(), joints2d[:, 1].min(), joints2d[:, 0].max(), joints2d[:, 1].max()]
        bbox = [x1, y1, x2, y2]        

        # for joint_i, (x,y) in enumerate(joints2d):
        #     x, y = int(x), int(y)
        #     if joints_vis[joint_i]:
        #         cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)
        #         cv2.putText(image, f'{joint_i}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        # for s,e in self.skeleton:
        #     if joints_vis[s] and joints_vis[e]:
        #         start_x,start_y = joints2d[s]
        #         end_x, end_y = joints2d[e]
        #         if start_x < 0 or end_x < 0:
        #             continue

        #         cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,0,0), 1)
        # cv2.imwrite(f'./check.png', image)

        out_sample = {}
        if 'segments' in sample:
            out_sample['segments'] = sample['segments']
            sample.pop('segments')

        sample.pop('image')
        sample.pop('joints2d')
        sample.pop('joints_vis')
        out_sample.update({
            'image': image,
            'joints_vis': np.expand_dims(joints_vis,0),
            'joints2d': np.expand_dims(joints2d,0),
            'bboxes': np.array(bbox)
        })
        out_sample.update(sample)
        return out_sample


# def cutout(
#     img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]], fill_value: Union[int, float] = 0
# ) -> np.ndarray:
#     # Make a copy of the input image since we don't want to modify it directly
#     img = img.copy()
#     for x1, y1, x2, y2 in holes:
#         img[y1:y2, x1:x2] = fill_value
#     return img


# @PIPELINES.register_module()
# class CoarseDropout(object):
#     """CoarseDropout of the rectangular regions in the image.

#     Args:
#         max_holes (int): Maximum number of regions to zero out.
#         max_height (int, float): Maximum height of the hole.
#         If float, it is calculated as a fraction of the image height.
#         max_width (int, float): Maximum width of the hole.
#         If float, it is calculated as a fraction of the image width.
#         min_holes (int): Minimum number of regions to zero out. If `None`,
#             `min_holes` is be set to `max_holes`. Default: `None`.
#         min_height (int, float): Minimum height of the hole. Default: None. If `None`,
#             `min_height` is set to `max_height`. Default: `None`.
#             If float, it is calculated as a fraction of the image height.
#         min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
#             set to `max_width`. Default: `None`.
#             If float, it is calculated as a fraction of the image width.

#         fill_value (int, float, list of int, list of float): value for dropped pixels.
#         mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
#             in mask. If `None` - mask is not affected. Default: `None`.

#     Targets:
#         image, mask, keypoints

#     Image types:
#         uint8, float32

#     Reference:
#     |  https://arxiv.org/abs/1708.04552
#     |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
#     |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
#     """

#     def __init__(
#         self,
#         max_holes: int = 8,
#         max_height: int = 8,
#         max_width: int = 8,
#         min_holes: Optional[int] = None,
#         min_height: Optional[int] = None,
#         min_width: Optional[int] = None,
#         fill_value: int = 0,
#         mask_fill_value: Optional[int] = None,
#         always_apply: bool = False,
#         p: float = 0.5,
#     ):
#         self.max_holes = max_holes
#         self.max_height = max_height
#         self.max_width = max_width
#         self.min_holes = min_holes if min_holes is not None else max_holes
#         self.min_height = min_height if min_height is not None else max_height
#         self.min_width = min_width if min_width is not None else max_width
#         self.fill_value = fill_value
#         self.mask_fill_value = mask_fill_value
#         if not 0 < self.min_holes <= self.max_holes:
#             raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))

#         self.check_range(self.max_height)
#         self.check_range(self.min_height)
#         self.check_range(self.max_width)
#         self.check_range(self.min_width)
#         self.p = p

#         if not 0 < self.min_height <= self.max_height:
#             raise ValueError(
#                 "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
#             )
#         if not 0 < self.min_width <= self.max_width:
#             raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

#     def check_range(self, dimension):
#         if isinstance(dimension, float) and not 0 <= dimension < 1.0:
#             raise ValueError(
#                 "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
#             )

#     def __call__(self, sample, context=None):
#         if np.random.random() > self.p:
#             return sample

#         params = self.get_params_dependent_on_targets(sample)
#         holes = params['holes']
#         image = cutout(sample['image'], holes, self.fill_value)
#         sample['image'] = image

#         return sample

#     def apply(
#         self,
#         img: np.ndarray,
#         fill_value: Union[int, float] = 0,
#         holes: Iterable[Tuple[int, int, int, int]] = (),
#         **params
#     ) -> np.ndarray:

#         return cutout(img, holes, fill_value)

#     def apply_to_mask(
#         self,
#         img: np.ndarray,
#         mask_fill_value: Union[int, float] = 0,
#         holes: Iterable[Tuple[int, int, int, int]] = (),
#         **params
#     ) -> np.ndarray:
#         if mask_fill_value is None:
#             return img
#         return cutout(img, holes, mask_fill_value)

#     def get_params_dependent_on_targets(self, params):
#         img = params["image"]
#         height, width = img.shape[:2]

#         holes = []
#         for _n in range(random.randint(self.min_holes, self.max_holes)):
#             if all(
#                 [
#                     isinstance(self.min_height, int),
#                     isinstance(self.min_width, int),
#                     isinstance(self.max_height, int),
#                     isinstance(self.max_width, int),
#                 ]
#             ):
#                 hole_height = random.randint(self.min_height, self.max_height)
#                 hole_width = random.randint(self.min_width, self.max_width)
#             elif all(
#                 [
#                     isinstance(self.min_height, float),
#                     isinstance(self.min_width, float),
#                     isinstance(self.max_height, float),
#                     isinstance(self.max_width, float),
#                 ]
#             ):
#                 hole_height = int(height * random.uniform(self.min_height, self.max_height))
#                 hole_width = int(width * random.uniform(self.min_width, self.max_width))
#             else:
#                 raise ValueError(
#                     "Min width, max width, \
#                     min height and max height \
#                     should all either be ints or floats. \
#                     Got: {} respectively".format(
#                         [
#                             type(self.min_width),
#                             type(self.max_width),
#                             type(self.min_height),
#                             type(self.max_height),
#                         ]
#                     )
#                 )

#             y1 = random.randint(0, height - hole_height)
#             x1 = random.randint(0, width - hole_width)
#             y2 = y1 + hole_height
#             x2 = x1 + hole_width
#             holes.append((x1, y1, x2, y2))

#         return {"holes": holes}

#     # @property
#     # def targets_as_params(self):
#     #     return ["image"]

#     # def _keypoint_in_hole(self, keypoint: KeypointType, hole: Tuple[int, int, int, int]) -> bool:
#     #     x1, y1, x2, y2 = hole
#     #     x, y = keypoint[:2]
#     #     return x1 <= x < x2 and y1 <= y < y2

#     # def apply_to_keypoints(
#     #     self, keypoints: Sequence[KeypointType], holes: Iterable[Tuple[int, int, int, int]] = (), **params
#     # ) -> List[KeypointType]:
#     #     result = set(keypoints)
#     #     for hole in holes:
#     #         for kp in keypoints:
#     #             if self._keypoint_in_hole(kp, hole):
#     #                 result.discard(kp)
#     #     return list(result)

#     # def get_transform_init_args_names(self):
#     #     return (
#     #         "max_holes",
#     #         "max_height",
#     #         "max_width",
#     #         "min_holes",
#     #         "min_height",
#     #         "min_width",
#     #         "fill_value",
#     #         "mask_fill_value",
#     #     )


@PIPELINES.register_module()
class AddExtInfo(object):
    def __init__(self):
        pass

    def __call__(self, sample, context=None):
        image = sample['image']
        image_h, image_w = image.shape[:2]
        sample.update({
            'has_joints': True,
            'has_segments': False,
            'segments': np.zeros((image_h, image_w), dtype=np.uint8),
        })

        return sample



@PIPELINES.register_module()
class WithPAFGenerator(object):
    def __init__(self, params, insize=384, stride=4, no_augment=True):
        self.params = params
        if isinstance(insize, tuple):
            self.insize = insize
        else:
            self.insize = (insize, insize)
        self.no_augment = no_augment
        self.stride = stride

    def generate_labels(self, img, poses, ignore_mask):
        if not self.no_augment:
            img, ignore_mask, poses = self.augment_data(img, ignore_mask, poses)
        resized_img, ignore_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape=self.insize)

        # 生成heatmap
        heatmaps = self.generate_heatmaps(resized_img, resized_poses, self.params['heatmap_sigma'], self.stride)
        # 生成paf
        pafs = self.generate_pafs(resized_img, resized_poses, self.params['paf_sigma'], self.stride) # params['paf_sigma']: 8
        # 生成忽略位置
        ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((4, 4))).astype('bool')
        return resized_img, pafs, heatmaps, ignore_mask

    def resize_data(self, img, ignore_mask, poses, shape):
        """resize img, mask and annotations"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))
        return resized_img, ignore_mask, poses

    def random_resize_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(axis=1)**0.5

        min_scale = self.params['min_box_size']/bbox_sizes.min()
        max_scale = self.params['max_box_size']/bbox_sizes.max()

        # print(len(bbox_sizes))
        # print('min: {}, max: {}'.format(min_scale, max_scale))

        min_scale = min(max(min_scale, self.params['min_scale']), 1)
        max_scale = min(max(max_scale, 1), self.params['max_scale'])

        # print('min: {}, max: {}'.format(min_scale, max_scale))

        scale = float((max_scale - min_scale) * random.random() + min_scale)
        shape = (round(w * scale), round(h * scale))

        # print(scale)

        resized_img, resized_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape)
        return resized_img, resized_mask, poses

    def random_rotate_img(self, img, mask, poses):
        h, w, _ = img.shape
        # degree = (random.random() - 0.5) * 2 * params['max_rotate_degree']
        degree = np.random.randn() / 3 * self.params['max_rotate_degree']
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)), w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(img, R, (int(bbox[0]+0.5), int(bbox[1]+0.5)), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=[127.5, 127.5, 127.5])
        rotate_mask = cv2.warpAffine(mask.astype('uint8')*255, R, (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0

        tmp_poses = np.ones_like(poses)
        tmp_poses[:, :, :2] = poses[:, :, :2].copy()
        tmp_rotate_poses = np.dot(tmp_poses, R.T)  # apply rotation matrix to the poses
        rotate_poses = poses.copy()  # to keep visibility flag
        rotate_poses[:, :, :2] = tmp_rotate_poses
        return rotate_img, rotate_mask, rotate_poses

    def augment_data(self, img, ignore_mask, poses):
        aug_img = img.copy()
        aug_img, ignore_mask, poses = self.random_resize_img(aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_rotate_img(aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_crop_img(aug_img, ignore_mask, poses)
        if np.random.randint(2):
            aug_img = self.distort_color(aug_img)
        return aug_img, ignore_mask, poses

    def distort_color(self, img):
        img_max = np.broadcast_to(np.array(255, dtype=np.uint8), img.shape[:-1])
        img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

        hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max), img_min) # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max), img_min) # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max), img_min) # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def random_crop_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        insize = min(self.insize[0], self.insize[1])
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox = random.choice(joint_bboxes)  # select a bbox randomly
        bbox_center = bbox[:2] + (bbox[2:] - bbox[:2])/2

        r_xy = np.random.rand(2)
        perturb = ((r_xy - 0.5) * 2 * self.params['center_perterb_max'])
        center = (bbox_center + perturb + 0.5).astype('i')

        crop_img = np.zeros((insize, insize, 3), 'uint8') + 127.5
        crop_mask = np.zeros((insize, insize), 'bool')

        offset = (center - (insize-1)/2 + 0.5).astype('i')
        offset_ = (center + (insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

        x1, y1 = (center - (insize-1)/2 + 0.5).astype('i')
        x2, y2 = (center + (insize-1)/2 + 0.5).astype('i')

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        x_from = -offset[0] if offset[0] < 0 else 0
        y_from = -offset[1] if offset[1] < 0 else 0
        x_to = insize - offset_[0] - 1 if offset_[0] >= 0 else insize - 1
        y_to = insize - offset_[1] - 1 if offset_[1] >= 0 else insize - 1

        crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
        crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[y1:y2+1, x1:x2+1].copy()

        poses[:, :, :2] -= offset
        return crop_img.astype('uint8'), crop_mask, poses

    def resize_data(self, img, ignore_mask, poses, shape):
        """resize img, mask and annotations"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))
        return resized_img, ignore_mask, poses

    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        #产生的就是一整张图的gaussian分布，只不过里中心点远的点非常非常小
        return gaussian_heatmap

    def generate_heatmaps(self, img, poses, heatmap_sigma, stride):
        joint_num = poses.shape[1]
        # sum_heatmap = np.zeros(img.shape[:-1])

        image_h, image_w = img.shape[:-1]
        t_image_h = int(image_h / stride)
        t_image_w = int(image_w / stride)
        t_poses = poses.copy()
        t_poses[:,:,:2] = poses[:, :, :2] * np.array((t_image_w, t_image_h)) / np.array((image_w, image_h))

        heatmaps = np.zeros((0,) + (t_image_h, t_image_w))
        for joint_index in range(joint_num):
            heatmap = np.zeros((t_image_h, t_image_w))
            for pose in t_poses:
                if pose[joint_index, 2] > 0:
                    # 可见点
                    jointmap = self.generate_gaussian_heatmap((t_image_h, t_image_w), pose[joint_index,:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    # sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]

            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        # bg_heatmap = 1 - sum_heatmap  # background channel
        # heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        '''
        We take the maximum of the confidence maps insteaof the average so that thprecision of close by peaks remains distinct, 
        as illus- trated in the right figure. At test time, we predict confidence maps (as shown in the first row of Fig. 4), 
        and obtain body part candidates by performing non-maximum suppression.
        At test time, we predict confidence maps (as shown in the first row of Fig. 4), 
        and obtain body part candidates by performing non-maximum suppression.
        '''
        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance 
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector) # 垂直分量
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # grid_x, grid_y用来遍历图上的每一个点
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        '''
        相当于遍历图上的每一个点，从这个点到joint_from的向量与unit_vector点乘
        两个向量点乘相当于取一个向量在另一个向量方向上的投影
        如果点乘大于0，那就可以判断这个点在不在这个躯干的方向上了，
        (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        这个限制条件是保证在与躯干水平的方向上，找出所有落在躯干范围内的点
        然而还要判断这个点离躯干的距离有多远
        '''
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width # paf_width : 8
        '''
        要判断这个点离躯干的距离有多远，只要拿与起始点的向量点乘垂直分量就可以了，
        所以这里的限制条件是paf_width, 不然一个手臂就无限粗了
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
        这个限制条件是保证在与躯干垂直的方向上，找出所有落在躯干范围内的点（这个躯干范围看来是手工定义的)
        '''
        paf_flag = horizontal_paf_flag & vertical_paf_flag # 合并两个限制条件
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)
        # constant_paf.shape : (2, 368, 368), 上面这一步就是把2维的unit_vector broadcast到所有paf_flag为true的点上去
        # constant_paf里面有368*368个点，每个点上有两个值，代表一个矢量
        # constant_paf里的这些矢量只会取两种值，要么是(0,0),要么是unit_vector的值
        '''最后，这个函数完成的是论文里公式8和公式9，相关说明也可以看论文这一段的描述'''
        return constant_paf

    def generate_pafs(self, img, poses, paf_sigma, stride):
        image_h, image_w = img.shape[:-1]
        t_image_h = int(image_h / stride)
        t_image_w = int(image_w / stride)
        t_poses = poses.copy()
        t_poses[:,:,:2] = poses[:, :, :2] * np.array((t_image_w, t_image_h)) / np.array((image_w, image_h))

        pafs = np.zeros((0,) + (t_image_h, t_image_w))
        for limb in self.params['limbs_point']:
            paf = np.zeros((2,) + (t_image_h, t_image_w))
            paf_flags = np.zeros(paf.shape) # for constant paf

            for pose in t_poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf((t_image_h, t_image_w, 3), joint_from[:2], joint_to[:2], paf_sigma) #[2,368,368]
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    '''
                    这个flags的作用是计数，在遍历了一张图上的所有人体之后，有的地方可能会有重叠，
                    比如说两个人的左手臂交织在一起，重叠的部分就累加了两次，
                    这里计数了之后，后面可以用来求均值
                    '''
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0] # 求均值
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def __call__(self, sample, context=None):
        img = sample['image']
        joints_vis = sample['joints_vis']
        if len(img.shape) == 2:
            img = np.stack([img,img,img], -1)

        if len(joints_vis.shape) != 3 or joints_vis.shape[0] == 0:
            return {
                'image': cv2.resize(img, (self.insize[0], self.insize[1])),
                'pafs': np.zeros((2*len(self.params['limbs_point']), self.insize[1]//self.stride, self.insize[0]//self.stride), dtype=np.float32),
                'heatmaps': np.zeros((33, self.insize[1]//self.stride, self.insize[0]//self.stride), dtype=np.float32),
                'ignore_mask': np.zeros((self.insize[1], self.insize[0]), dtype=np.float32),
                'heatmap_valid_mask': np.ones((33, self.insize[1]//self.stride, self.insize[0]//self.stride), dtype=np.float32),
                'paf_valid_mask':  np.ones((2*len(self.params['limbs_point']), self.insize[1]//self.stride, self.insize[0]//self.stride), dtype=np.float32)
            }

        joints_islabel = joints_vis[:,:, 1][0]
        poses = sample['joints2d']

        poses = np.concatenate([poses, joints_vis[:,:, 0:1] ], axis=-1)
        joint_num = joints_vis.shape[1]
        h,w = img.shape[:2]
        ignore_mask = np.zeros((h,w), dtype=np.uint8)
        resized_img, pafs, heatmaps, ignore_mask = self.generate_labels(img, poses, ignore_mask)
        ignore_mask = ignore_mask.astype('f')

        # # debug heatmap
        # for joint_i in range(33):
        #     joint_i_img = img.copy()
        #     joint_i_img_h, joint_i_img_w = joint_i_img.shape[:2]
        #     resized_joint_i_heatmap = cv2.resize(heatmaps[joint_i], (joint_i_img_w, joint_i_img_h))
        #     joint_i_img = joint_i_img * (1-np.expand_dims(resized_joint_i_heatmap, -1)) + np.expand_dims(resized_joint_i_heatmap, -1)*np.array([0,0,255]).reshape((1,1,3))
        #     cv2.imwrite(f'./{joint_i}.png', joint_i_img.astype(np.uint8))
        heatmap_h, heatmap_w = heatmaps.shape[1:]
        heatmap_valid_mask = np.zeros((joint_num, heatmap_h, heatmap_w), dtype=np.float32)
        heatmap_valid_mask[joints_islabel != 0] = 1

        invalid_joint_ids = np.where(joints_islabel == 0)[0]
        # # debug paf
        # for limb_i in range(len(self.params['limbs_point'])):
        #     if self.params['limbs_point'][limb_i][0] not in invalid_joint_ids and \
        #         self.params['limbs_point'][limb_i][1] not in invalid_joint_ids:
        #         limb_i_img = img.copy()
        #         resized_limb_i_paf = cv2.resize(pafs[2*limb_i], (limb_i_img.shape[1], limb_i_img.shape[0]))
        #         limb_i_img = limb_i_img * (1-np.expand_dims(resized_limb_i_paf, -1)) + np.expand_dims(resized_limb_i_paf, -1)*np.array([0,0,255]).reshape((1,1,3))
        #         cv2.imwrite(f'./limb_x_{limb_i}.png', limb_i_img.astype(np.uint8))
        #         limb_i_img = img.copy()
        #         resized_limb_i_paf = cv2.resize(pafs[2*limb_i+1], (limb_i_img.shape[1], limb_i_img.shape[0]))
        #         limb_i_img = limb_i_img * (1-np.expand_dims(resized_limb_i_paf, -1)) + np.expand_dims(resized_limb_i_paf, -1)*np.array([0,255,0]).reshape((1,1,3))
        #         cv2.imwrite(f'./limb_y_{limb_i}.png', limb_i_img.astype(np.uint8))

        paf_h, paf_w = pafs.shape[1:]
        paf_valid_maks = np.ones((2*len(self.params['limbs_point']), paf_h, paf_w), dtype=np.float32)
        for limb_i in range(len(self.params['limbs_point'])):
            if self.params['limbs_point'][limb_i][0] in invalid_joint_ids or \
                self.params['limbs_point'][limb_i][1] in invalid_joint_ids:
                paf_valid_maks[2*limb_i:2*(limb_i+1)] = 0

        return {
            'image': resized_img,
            'pafs': pafs,
            'heatmaps': heatmaps,
            'ignore_mask': ignore_mask,
            'heatmap_valid_mask': heatmap_valid_mask,
            'paf_valid_mask': paf_valid_maks
        }