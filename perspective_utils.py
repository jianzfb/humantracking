#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from aipack.products.ai.mvdetect3d.perspective_crop.camera_model import (
    CameraToUV,
    UVToUnitSphere,
    distort2plane,
    plane2sphere,
)


def pad_image_to_shape(img, shape, *, return_padding=False):
    shape = list(shape[:2])
    if img.ndim > 2:
        shape.extend(img.shape[2:])
    shape = tuple(shape)

    h, w = img.shape[:2]
    assert w <= shape[1] and h <= shape[0]
    pad_width = shape[1] - w
    pad_height = shape[0] - h

    pad_w0 = pad_width // 2
    pad_w1 = shape[1] - (pad_width - pad_w0)
    pad_h0 = pad_height // 2
    pad_h1 = shape[0] - (pad_height - pad_h0)

    ret = np.zeros(shape, dtype=img.dtype)
    ret[pad_h0:pad_h1, pad_w0:pad_w1] = img
    if return_padding:
        return ret, (pad_h0, pad_w0)
    else:
        return ret


def resize_preserve_aspect_ratio(img, target_shape, interpolation=None, padding=True):
    assert img.size != 0

    h, w = img.shape[:2]
    th, tw = target_shape[:2]

    w_ratio = tw / float(w)
    h_ratio = th / float(h)

    ratio = min(w_ratio, h_ratio)

    cw = int(max(1, w * ratio))
    ch = int(max(1, h * ratio))
    if interpolation is not None:
        img = cv2.resize(img, (cw, ch), interpolation=interpolation)
    else:
        img = cv2.resize(img, (cw, ch))

    if padding:
        return pad_image_to_shape(img, target_shape)
    else:
        return img


def pixel2sphere(points_pixel, cu, cv, fu_reciprocal, fv_reciprocal, camera_distortions, camera_epsilon):
    """
    Args:
        pixel_points (array): [N, 2], uv

    Returns:
        sphere_points (array): [N, 3], xyz
    """
    assert points_pixel.shape[1] == 2

    points_distort = np.asarray(
        (
            fu_reciprocal * (points_pixel[:, 0] - cu),
            fv_reciprocal * (points_pixel[:, 1] - cv),
        )
    )
    points_plane = distort2plane(points_distort, camera_distortions)
    points_sphere = plane2sphere(points_plane, camera_epsilon)
    return points_sphere
    # return points_plane


def get_perspective_coords(K_virt, R_virt2orig, camera_model, crop_pixel_size_wh):
    # start = time.time()
    batch_size = K_virt.shape[0]
    device = K_virt.device

    xs = torch.linspace(0, 1, crop_pixel_size_wh[0]).to(device)
    ys = torch.linspace(0, 1, crop_pixel_size_wh[1]).to(device)

    rs, cs = torch.meshgrid([xs, ys])  # for pytorch >0.4 instead of following two lines
    zs = torch.ones(rs.shape).to(device)  # init homogeneous coordinate to 1
    pv = torch.stack([rs, cs, zs])

    # same input grid for all batch elements, expand along batch dimension
    grid = pv.unsqueeze(0).expand([batch_size, 3, crop_pixel_size_wh[0], crop_pixel_size_wh[1]])

    # linearize the 2D grid to a single dimension, to apply transformation
    patch_coords = grid.view([batch_size, 3, -1])

    # baseline version
    K_virt_inv = torch.inverse(K_virt)
    orig_points = torch.bmm(torch.bmm(R_virt2orig, K_virt_inv), patch_coords)
    orig_coords = CameraToUV(
        orig_points.transpose(1, 2),
        camera_model.camera_intrinsics,
        camera_model.camera_epsilon,
        camera_model.camera_distortions,
    )
    orig_coords = orig_coords.transpose(1, 2)
    orig_coords = orig_coords.view(batch_size, 2, crop_pixel_size_wh[0], crop_pixel_size_wh[1])

    # the sampling function assumes the position information on the last dimension
    orig_coords = orig_coords.permute([0, 3, 2, 1])
    return orig_coords


def perspective_grid(
    P_virt2orig,
    R_virt2orig,
    K_virt,
    camera_model,
    positions_px,
    image_pixel_size,
    crop_pixel_size_wh,
    transform_to_pytorch=False,
):
    orig_coords = get_perspective_coords(K_virt, R_virt2orig, camera_model, crop_pixel_size_wh)
    border_coords = get_perspective_coords(K_virt, R_virt2orig, camera_model, torch.Size([3, 3]))

    # the transformed points will be in pixel coordinates ranging from 0 up to the image width/height (unmapped from the original intrinsics matrix)
    # but the pytorch grid_sample function assumes it in -1,..,1; the direction is already correct (assuming negative y axis, which is also assumed by bytorch)
    if transform_to_pytorch:
        orig_coords /= image_pixel_size.view([1, 1, 1, 2])  # map to 0..1
        orig_coords *= 2  # to 0...2
        orig_coords -= 1  # to -1...1

    return orig_coords, border_coords


def pcl_transforms(
    bbox_pos_img,
    bbox_size_img,
    camera_model,
    R_rand=None,
    focal_at_image_plane=False,
    slant_compensation=False,
    rectangular_images=False,
    internal_call=False,
):
    # K_inv = torch.inverse(K)
    # get target position from image coordinates (normalized pixels)
    # p_position = bmm_homo(K_inv, bbox_pos_img)
    # p_position = pixel2sphere(bbox_pos_img, camera_model.cu, camera_model.cv, camera_model.fu_reciprocal, camera_model.fv_reciprocal, camera_model.camera_distortions, camera_model.camera_epsilon)
    p_position = UVToUnitSphere(
        bbox_pos_img.numpy().reshape(-1, 1),
        camera_model.cu,
        camera_model.cv,
        camera_model.fu_reciprocal,
        camera_model.fv_reciprocal,
        camera_model.camera_distortions,
        camera_model.camera_epsilon,
    )
    p_position = (p_position / p_position[2]).reshape(-1, 3)

    K_equal_effect = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    K_equal_effect[:, 0, 0] = (bbox_pos_img[:, 0] - camera_model.cu) / p_position[:, 0]
    K_equal_effect[:, 1, 1] = (bbox_pos_img[:, 1] - camera_model.cv) / p_position[:, 1]
    K_equal_effect[:, 0, 2] = camera_model.cu
    K_equal_effect[:, 1, 2] = camera_model.cv

    if abs(p_position[:, 0]) < 0.02 and abs(p_position[:, 1]) < 0.02:  # 2cm
        K_equal_effect[:, 0, 0] = 1 / (camera_model.fu_reciprocal * camera_model.camera_epsilon)
        K_equal_effect[:, 1, 1] = 1 / (camera_model.fv_reciprocal * camera_model.camera_epsilon)
    elif abs(p_position[:, 0]) < 0.02:
        K_equal_effect[:, 0, 0] = K_equal_effect[:, 1, 1]
    elif abs(p_position[:, 1]) < 0.02:
        K_equal_effect[:, 1, 1] = K_equal_effect[:, 0, 0]

    # get rotation from orig to new coordinate frame
    R_virt2orig = virtualCameraRotationFromPosition(p_position).type(torch.float32)
    if R_rand is not None:
        R_virt2orig = torch.matmul(R_virt2orig, torch.inverse(R_rand))

    # determine target frame
    K_virt = bK_virt(
        p_position,
        K_equal_effect,
        camera_model.camera_epsilon,
        bbox_size_img,
        focal_at_image_plane,
        slant_compensation,
        maintain_aspect_ratio=True,
        rectangular_images=rectangular_images,
    )

    # K_virt_inv = torch.inverse(K_virt)
    # projective transformation orig to virtual camera
    # P_virt2orig = torch.bmm(K, torch.bmm(R_virt2orig, K_virt_inv))
    P_virt2orig = None

    if not internal_call:
        return P_virt2orig, R_virt2orig, K_virt
    else:
        R_orig2virt = torch.inverse(R_virt2orig)
        P_orig2virt = torch.inverse(P_virt2orig)
        return P_virt2orig, R_virt2orig, K_virt, R_orig2virt, P_orig2virt


# def pcl_transforms_2d(
#     pose2d, bbox_pos_img, bbox_size_img, K, focal_at_image_plane=True, slant_compensation=True, rectangular_images=False
# ):
#     # create canonical labels
#     batch_size = pose2d.shape[0]
#     num_joints = pose2d.shape[1]
#     ones = torch.ones([batch_size, num_joints, 1])

#     P_virt2orig, R_virt2orig, K_virt, R_orig2virt, P_orig2virt = pcl_transforms(
#         bbox_pos_img, bbox_size_img, K, focal_at_image_plane, slant_compensation, rectangular_images, internal_call=True
#     )

#     # Vector manipulation to use torch.bmm and transform original 2D pose to virtual camera coordinates
#     P_orig2virt = P_orig2virt.unsqueeze(1).repeat(1, num_joints, 1, 1)
#     P_orig2virt = P_orig2virt.view(batch_size * num_joints, 3, 3)

#     canonical_2d_pose = torch.cat((pose2d, ones), dim=-1).unsqueeze(-1)
#     canonical_2d_pose = canonical_2d_pose.view(batch_size * num_joints, 3, 1)
#     PCL_canonical_2d_pose = torch.bmm(P_orig2virt, canonical_2d_pose)
#     PCL_canonical_2d_pose = PCL_canonical_2d_pose.squeeze(-1).view(batch_size, num_joints, -1)

#     # Convert from homogeneous coordinate by dividing x and y by z
#     virt_2d_pose = torch.div(PCL_canonical_2d_pose[:, :, :-1], PCL_canonical_2d_pose[:, :, -1].unsqueeze(-1))
#     return virt_2d_pose, R_virt2orig, P_virt2orig


def virtPose2CameraPose(virt_pose, R_virt2orig, batch_size, num_joints):
    # for input 3d pose
    R_virt2orig = R_virt2orig.unsqueeze(1).repeat(1, num_joints, 1, 1)

    virt_pose = virt_pose.view(batch_size * num_joints, 3, 1)
    R_virt2orig = R_virt2orig.view(batch_size * num_joints, 3, 3)

    # Matrix Multiplication
    camera_pose = torch.bmm(R_virt2orig, virt_pose)
    camera_pose = camera_pose.squeeze(-1).view(batch_size, num_joints, -1)

    return camera_pose


def bK_virt(
    p_position,
    K,
    camera_epsilon,
    bbox_size_img,
    focal_at_image_plane,
    slant_compensation,
    maintain_aspect_ratio=True,
    rectangular_images=False,
):
    batch_size = bbox_size_img.shape[0]
    p_length = torch.norm(p_position, dim=1, keepdim=True)
    focal_length_factor = 1
    if focal_at_image_plane:
        focal_length_factor *= p_length
    if slant_compensation:
        sx = 1.0 / torch.sqrt(p_position[:, 0] ** 2 + p_position[:, 2] ** 2)  # this is cos(phi)
        sy = torch.sqrt(p_position[:, 0] ** 2 + 1) / torch.sqrt(
            p_position[:, 0] ** 2 + p_position[:, 1] ** 2 + 1
        )  # this is cos(theta)
        bbox_size_img = bbox_size_img * torch.stack([sx, sy], dim=1)

    if not rectangular_images:
        if maintain_aspect_ratio:
            max_width, _ = torch.max(bbox_size_img, dim=-1, keepdims=True)
            bbox_size_img = torch.cat([max_width, max_width], dim=-1)
        f_orig = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        f_compensated = (
            focal_length_factor * f_orig / bbox_size_img
        )  # dividing by the target bbox_size_img will make the coordinates normalized to 0..1, as needed for the perspective grid sample function; an alternative would be to make the grid_sample operate on pixel coordinates
        K_virt = torch.zeros([batch_size, 3, 3], dtype=torch.float).to(f_compensated.device)
        K_virt[:, 2, 2] = 1
        # Note, in unit image coordinates ranging from 0..1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:, :2, 2] = 0.5
        return K_virt
    else:
        f_orig = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        f_re_scaled = f_orig / bbox_size_img
        if maintain_aspect_ratio:
            min_factor, _ = torch.min(f_re_scaled, dim=-1, keepdims=True)
            f_re_scaled = torch.cat([min_factor, min_factor], dim=-1)
        f_compensated = focal_length_factor * f_re_scaled
        K_virt = torch.zeros([batch_size, 3, 3], dtype=torch.float).to(f_compensated.device)
        K_virt[:, 2, 2] = 1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:, :2, 2] = 0.5
        return K_virt


def virtualCameraRotationFromPosition(position):
    x, y, _ = position[:, 0], (position[:, 1]), position[:, 2]
    n1x = torch.sqrt(1 + x**2)
    d1x = 1 / n1x
    d1xy = 1 / torch.sqrt(1 + x**2 + y**2)
    d1xy1x = 1 / torch.sqrt((1 + x**2 + y**2) * (1 + x**2))
    R_virt2orig = torch.stack(
        [d1x, -x * y * d1xy1x, x * d1xy, 0 * x, n1x * d1xy, y * d1xy, -x * d1x, -y * d1xy1x, 1 * d1xy], dim=1
    ).reshape([-1, 3, 3])
    return R_virt2orig


def bmm_homo(K_inv, bbox_center_img):
    batch_size = bbox_center_img.shape[0]
    ones = torch.ones([batch_size, 1], dtype=torch.float).to(bbox_center_img.device)
    bbox_center_px_homo = torch.cat([bbox_center_img, ones], dim=1).reshape([batch_size, 3, 1])
    cam_pos = torch.bmm(K_inv, bbox_center_px_homo).view(batch_size, -1)
    return cam_pos


def K_px2K_torch(K_px, img_w_h):
    K_torch = K_px.clone()
    K_torch[:, 0, 0] = K_px[:, 0, 0] * 2 / img_w_h[0]  # spread out from 0..w to -1..1
    K_torch[:, 1, 1] = K_px[:, 1, 1] * 2 / img_w_h[1]  # spread out from 0..h to -1..1
    K_torch[:, 0, 2] = K_px[:, 0, 2] * 2 / img_w_h[0] - 1  # move image origin bottom left corner to to 1/2 image width
    K_torch[:, 1, 2] = K_px[:, 1, 2] * 2 / img_w_h[1] - 1  # move image origin to 1/2 image width

    K_torch[:, 1, 1] *= -1  # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
    K_torch[:, 1, 2] *= -1  # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
    return K_torch


def transform_joint(joint2d, R_virt2orig, K_virt, camera_model):
    joint_3d_sphere = UVToUnitSphere(
        joint2d.T,
        camera_model.cu,
        camera_model.cv,
        camera_model.fu_reciprocal,
        camera_model.fv_reciprocal,
        camera_model.camera_distortions,
        camera_model.camera_epsilon,
    )
    joint_3d_sphere = torch.matmul(R_virt2orig.T, joint_3d_sphere)
    joint2d_proj = torch.matmul(K_virt, joint_3d_sphere)
    joint2d_proj /= joint2d_proj[2:3, :].clone()
    return joint2d_proj[:2, :].T


def build_map_table(distortion, intri, coords):
    mx2_u = coords[0, :] * coords[0, :]
    my2_u = coords[1, :] * coords[1, :]
    mxy_u = coords[0, :] * coords[1, :]
    rho2_u = mx2_u + my2_u
    rad_dist_u = distortion[0] * rho2_u + distortion[1] * rho2_u * rho2_u
    distort_x = (
        coords[0, :] + coords[0, :] * rad_dist_u + 2.0 * distortion[2] * mxy_u + distortion[3] * (rho2_u + 2.0 * mx2_u)
    )
    distort_y = (
        coords[1, :] + coords[1, :] * rad_dist_u + 2.0 * distortion[3] * mxy_u + distortion[2] * (rho2_u + 2.0 * my2_u)
    )
    uint_z = torch.ones_like(distort_x)
    out = torch.stack([distort_x, distort_y, uint_z], dim=0)
    out = torch.matmul(intri, out)
    return out


def pcl_virt2orig_jagged(rkinv_virt, xi, lut_table, coords, vmin=-0.285, vmax=0.285, grid_x=720, grid_y=720):
    out = torch.bmm(rkinv_virt, coords)
    out = out.transpose(1, 2)
    out = F.normalize(out, p=2, dim=2)
    out[:, :, 2] += xi
    out = out[:, :, :2] / out[:, :, 2:]
    # look up the table
    sx = (vmax - vmin) / (grid_x - 1)
    sy = (vmax - vmin) / (grid_y - 1)
    index_x = torch.floor((out[:, :, 0] - vmin) / sx + 0.5).long()
    index_y = torch.floor((out[:, :, 1] - vmin) / sy + 0.5).long()
    index = index_x + index_y * grid_x
    index = torch.clamp(index, min=0, max=grid_x * grid_y - 1).long()
    coord_x = lut_table[0, index]
    coord_y = lut_table[1, index]
    return coord_x, coord_y


def initialze_grid_xy(num_x=256, num_y=256, vmin=-0.285, vmax=0.285):
    x = torch.linspace(vmin, vmax, num_x)
    y = torch.linspace(vmin, vmax, num_y)
    gy, gx = torch.meshgrid([y, x])
    gz_sq = 1.0 - gx**2 - gy**2
    valid = gz_sq >= 0.0
    gz = torch.zeros_like(gz_sq)
    gz[valid] = torch.sqrt(gz_sq[valid])
    grid_xyz = torch.stack([gx, gy, gz], dim=0)
    return grid_xyz.view(3, -1)


def perspective_grid_lut(
    R_virt2orig,
    K_virt,
    camera_model,
    crop_pixel_size_wh,
    image_pixel_size,
    grid_x=720,
    grid_y=720,
    vmin=-0.285,
    vmax=0.285,
    transform_to_pytorch=True,
):
    device = K_virt.device
    batch_size = K_virt.shape[0]
    grid_coords = initialze_grid_xy(grid_x, grid_y, vmin, vmax)
    lut_table = build_map_table(
        torch.tensor(camera_model.camera_distortions).float().to(device),
        torch.tensor(camera_model.camera_intrinsics).float().to(device),
        grid_coords.to(device),
    )
    K_virt_inv = torch.inverse(K_virt)
    RK_inv_virt = torch.bmm(R_virt2orig, K_virt_inv)

    xs = torch.linspace(0, 1, crop_pixel_size_wh[0]).to(device)
    ys = torch.linspace(0, 1, crop_pixel_size_wh[1]).to(device)
    cs, rs = torch.meshgrid([ys, xs])
    zs = torch.ones(rs.shape).to(device)
    pv = torch.stack([rs, cs, zs])
    grid = pv.unsqueeze(0).expand([batch_size, 3, crop_pixel_size_wh[0], crop_pixel_size_wh[1]])
    patch_coords = grid.view([batch_size, 3, -1])

    coords_x, coords_y = pcl_virt2orig_jagged(
        RK_inv_virt,
        torch.tensor(camera_model.camera_epsilon).float().to(device),
        lut_table,
        patch_coords,
        vmin=-0.285,
        vmax=0.285,
        grid_x=720,
        grid_y=720,
    )

    orig_coords = torch.stack([coords_x, coords_y], dim=1)
    orig_coords = orig_coords.view(batch_size, 2, crop_pixel_size_wh[0], crop_pixel_size_wh[1])
    orig_coords = orig_coords.permute([0, 3, 2, 1])
    orig_coords = orig_coords.transpose(2, 1)
    if transform_to_pytorch:
        orig_coords /= image_pixel_size.view([1, 1, 1, 2])  # map to 0..1
        orig_coords *= 2  # to 0...2
        orig_coords -= 1  # to -1...1

    border_coords = get_perspective_coords(K_virt, R_virt2orig, camera_model, torch.Size([3, 3]))

    return orig_coords, border_coords


def perspective_crop_image(img, bbox, joint2d_ori, crop_size, camera_model, R_rand=None, rand_scale=None):
    crop_resolution_px = torch.Size(crop_size)
    image_resolution_px = torch.FloatTensor([img.shape[1], img.shape[0]])

    # object center position
    positions_px = torch.FloatTensor([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])
    scales_px = torch.tensor([[bbox[2] - bbox[0], bbox[3] - bbox[1]]])

    # convert to pytorch positions (-1..1 with y pointing downwards)
    # positions = 2 * positions_px / image_resolution_px - 1

    # nr_sample = len(positions_px)
    # select the same camera for all frames
    # Ks_px = Ks_px.unsqueeze(0).expand([nr_sample, 3, 3])
    # Ks = K_px2K_torch(Ks_px, image_resolution_px, camera_model.camera_epsilon)

    # run PCL
    P_virt2orig, R_virt2orig, K_virt = pcl_transforms(positions_px, scales_px, camera_model, R_rand)
    # print(K_virt.shape)
    if rand_scale is not None:
        K_virt[:, 0, 0] *= rand_scale
        K_virt[:, 1, 1] *= rand_scale

    joint2d_proc = transform_joint(joint2d_ori, R_virt2orig[0], K_virt[0], camera_model)

    # generate perspective grid
    orig_coords, border_coords = perspective_grid(
        P_virt2orig,
        R_virt2orig,
        K_virt,
        camera_model,
        positions_px,
        image_resolution_px,
        crop_resolution_px,
        transform_to_pytorch=True,
    )

    # load image
    img_orig = torch.FloatTensor(img) / 256

    img_torch = img_orig.permute([2, 0, 1])

    # sample each image separately
    # start = time.time()
    img_crop_perspective = F.grid_sample(img_torch.unsqueeze(0), orig_coords, align_corners=True)[0].permute([1, 2, 0])
    img_crop = (img_crop_perspective.data.numpy() * 255).astype(np.uint8)

    # orig_coords_lut, border_coords_lut = perspective_grid_lut(
    #     R_virt2orig,
    #     K_virt,
    #     camera_model,
    #     crop_resolution_px,
    #     image_resolution_px,
    #     grid_x=720,
    #     grid_y=720,
    #     vmin=-0.285,
    #     vmax=0.285,
    #     transform_to_pytorch=True)

    # img_crop_perspective_lut = F.grid_sample(img_torch.unsqueeze(0), orig_coords_lut, align_corners=True)[0].permute([1, 2, 0])
    # img_crop_lut = (img_crop_perspective_lut.data.numpy() * 255).astype(np.uint8)

    # border_coords_np = border_coords.reshape(-1, 2).data.numpy().astype(np.int32)
    return img_crop, joint2d_proc.numpy(), R_virt2orig[0].numpy(), K_virt[0].numpy()
    # return img_crop_lut, joint2d_proc.numpy(), R_virt2orig[0].numpy(), K_virt[0].numpy()
