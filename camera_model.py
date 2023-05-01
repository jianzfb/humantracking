import numpy as np
import numpy as jnp
import torch


def projection(points, extrinsics):
    homo_points = jnp.vstack([points, [1.0] * points.shape[1]])
    ext_homo_points = extrinsics @ homo_points
    return ext_homo_points[:-1, :]


def distort2plane(points_distort, distortions, max_iter_num=5):
    points_plane = points_distort.copy()
    k1, k2, k3, k4 = distortions
    # undistort
    u, v = points_plane
    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    rd = k1 * r2 + k2 * r2 * r2
    points_uv_new = np.asarray(
        (
            u + u * rd + 2 * k3 * uv + k4 * (r2 + 2 * u2),
            v + v * rd + 2 * k4 * uv + k3 * (r2 + 2 * v2),
        )
    )
    return points_uv_new


def plane2sphere(points_plane, epsilon):
    u = points_plane[0]
    v = points_plane[1]

    rho2_d = u * u + v * v
    temp = 1 + (1 - epsilon * epsilon) * rho2_d
    d1 = epsilon * (rho2_d + 1)
    d2 = epsilon + np.sqrt(np.maximum(temp, 0))
    d = 1 - d1 / d2

    points_sphere = np.asarray((u, v, d)).T
    points_sphere /= np.linalg.norm(points_sphere[np.newaxis, :], axis=1)
    return points_sphere


def camera2pixel(camera_points, epsilon, distortions, intrinsics):
    x, y, z = camera_points
    x /= z
    y /= z

    # plane2distort
    k1, k2, k3, k4 = distortions
    x2 = x * x
    xy = x * y
    y2 = y * y
    r2 = x2 + y2
    rd = k1 * r2 + k2 * r2 * r2
    dx = x + x * rd + 2.0 * k3 * xy + k4 * (r2 + 2.0 * x2)
    dy = y + y * rd + 2.0 * k4 * xy + k3 * (r2 + 2.0 * y2)
    distort_homo_points = torch.vstack([dx, dy, torch.ones_like(dx)])

    # distort2pixel
    intri = torch.from_numpy(intrinsics).float()
    pixel_homo_points = torch.matmul(intri, distort_homo_points)
    pixel_points = pixel_homo_points[:-1, :]

    return pixel_points


def distort(inputP, cCmaD):
    _k1 = cCmaD[0]
    _k2 = cCmaD[1]
    _p1 = cCmaD[2]
    _p2 = cCmaD[3]

    J = np.zeros((2, 2), np.float32)
    y = inputP
    mx2_u = y[0] * y[0]
    my2_u = y[1] * y[1]
    mxy_u = y[0] * y[1]
    rho2_u = mx2_u + my2_u

    rad_dist_u = _k1 * rho2_u + _k2 * rho2_u * rho2_u

    J[0, 0] = 1 + rad_dist_u + _k1 * 2.0 * mx2_u + _k2 * rho2_u * 4 * mx2_u + 2.0 * _p1 * y[1] + 6 * _p2 * y[0]

    J[1, 0] = _k1 * 2.0 * y[0] * y[1] + _k2 * 4 * rho2_u * y[0] * y[1] + _p1 * 2.0 * y[0] + 2.0 * _p2 * y[1]

    J[0, 1] = J[1, 0]
    J[1, 1] = 1 + rad_dist_u + _k1 * 2.0 * my2_u + _k2 * rho2_u * 4 * my2_u + 6 * _p1 * y[1] + 2.0 * _p2 * y[0]

    tem_x = y[0] + y[0] * rad_dist_u + 2.0 * _p1 * mxy_u + _p2 * (rho2_u + 2.0 * mx2_u)
    tem_y = y[1] + y[1] * rad_dist_u + 2.0 * _p2 * mxy_u + _p1 * (rho2_u + 2.0 * my2_u)

    return (tem_x, tem_y), J


def distort_array(inputP, cCmaD):
    _k1 = cCmaD[0]
    _k2 = cCmaD[1]
    _p1 = cCmaD[2]
    _p2 = cCmaD[3]

    J = np.zeros((inputP.shape[1], 2, 2), np.float32)
    y = inputP
    mx2_u = y[0] * y[0]
    my2_u = y[1] * y[1]
    mxy_u = y[0] * y[1]
    rho2_u = mx2_u + my2_u

    rad_dist_u = _k1 * rho2_u + _k2 * rho2_u * rho2_u

    J[:, 0, 0] = 1 + rad_dist_u + _k1 * 2.0 * mx2_u + _k2 * rho2_u * 4 * mx2_u + 2.0 * _p1 * y[1] + 6 * _p2 * y[0]

    J[:, 1, 0] = _k1 * 2.0 * y[0] * y[1] + _k2 * 4 * rho2_u * y[0] * y[1] + _p1 * 2.0 * y[0] + 2.0 * _p2 * y[1]

    J[:, 0, 1] = J[:, 1, 0]
    J[:, 1, 1] = 1 + rad_dist_u + _k1 * 2.0 * my2_u + _k2 * rho2_u * 4 * my2_u + 6 * _p1 * y[1] + 2.0 * _p2 * y[0]

    tem_x = y[0] + y[0] * rad_dist_u + 2.0 * _p1 * mxy_u + _p2 * (rho2_u + 2.0 * mx2_u)
    tem_y = y[1] + y[1] * rad_dist_u + 2.0 * _p2 * mxy_u + _p1 * (rho2_u + 2.0 * my2_u)

    return np.concatenate([tem_x.reshape(1, -1), tem_y.reshape(1, -1)], axis=0), J


def undistort(inputP, cCmaD):
    n = 5
    ybar = inputP
    for i in range(n):
        y_tmp = ybar
        y_tmp, F = distort(y_tmp, cCmaD)
        e_x = inputP[0] - y_tmp[0]
        e_y = inputP[1] - y_tmp[1]
        e = (e_x, e_y)
        mat_ = np.matmul(np.linalg.inv(np.matmul(np.transpose(F), F)), np.transpose(F))

        du_x = mat_[0, 0] * e[0] + mat_[0, 1] * e[1]
        du_y = mat_[1, 0] * e[0] + mat_[1, 1] * e[1]

        tem_x = ybar[0] + du_x
        tem_y = ybar[1] + du_y
        ybar = (tem_x, tem_y)

        if (e[0] * e[0] + e[1] * e[1]) < 1e-15:
            break

    return ybar


def undistort_array(inputP, cCmaD):
    n = 5
    ybar = inputP
    for i in range(n):
        y_tmp = ybar
        y_tmp, F = distort_array(y_tmp, cCmaD)
        e_x = inputP[0] - y_tmp[0]
        e_y = inputP[1] - y_tmp[1]
        e = (e_x, e_y)
        mat_ = np.matmul(np.linalg.inv(np.matmul(np.transpose(F, (0, 2, 1)), F)), np.transpose(F, (0, 2, 1)))

        du_x = mat_[:, 0, 0] * e[0] + mat_[:, 0, 1] * e[1]
        du_y = mat_[:, 1, 0] * e[0] + mat_[:, 1, 1] * e[1]

        tem_x = ybar[0] + du_x
        tem_y = ybar[1] + du_y
        ybar = np.concatenate([tem_x.reshape(1, -1), tem_y.reshape(1, -1)], axis=0)

        if (e[0] * e[0] + e[1] * e[1]).min() < 1e-15:
            break

    return ybar


def UVToUnitSphere(inputP, _cu, _cv, _recip_fu, _recip_fv, cCmaD, _xi):
    # Unproject...
    oututP = np.zeros((3, inputP.shape[1]))

    inputP_norm = np.zeros((2, inputP.shape[1]))
    inputP_norm[0] = _recip_fu * (inputP[0] - _cu)
    inputP_norm[1] = _recip_fv * (inputP[1] - _cv)

    p2Norm = undistort_array(inputP_norm, cCmaD)
    outPoint_x = p2Norm[0]
    outPoint_y = p2Norm[1]

    rho2_d = outPoint_x * outPoint_x + outPoint_y * outPoint_y

    _one_over_xixi_m_1 = 1.0 / (_xi * _xi - 1.0)
    isT = (_xi < 1.0 or _xi == 1.0) or ((rho2_d <= _one_over_xixi_m_1).sum() == rho2_d.shape[0])
    if not isT:
        return None

    outPoint_z = 1 - _xi * (rho2_d + 1) / (_xi + np.sqrt(1 + (1 - _xi * _xi) * rho2_d))

    oututP[0, :] = outPoint_x
    oututP[1, :] = outPoint_y
    oututP[2, :] = outPoint_z

    return torch.tensor(oututP).type(torch.float32)


def distort_before_projection(inputP, cCmaD):
    """
    Input:
        inputP: normalize 3d point [bs,N,3]
        cCmaD: camera distortion coefficients [4]
    Return:
        outputP: inputP after distortion [bs,N,3]
    """
    _k1 = cCmaD[0]
    _k2 = cCmaD[1]
    _p1 = cCmaD[2]
    _p2 = cCmaD[3]

    p = inputP
    p = p / p[:, :, -1].unsqueeze(-1)
    mx2_u = p[:, :, 0] * p[:, :, 0]
    my2_u = p[:, :, 1] * p[:, :, 1]
    mxy_u = p[:, :, 0] * p[:, :, 1]
    rho2_u = mx2_u + my2_u
    rad_dist_u = _k1 * rho2_u + _k2 * rho2_u * rho2_u

    distort_x = p[:, :, 0] + p[:, :, 0] * rad_dist_u + 2.0 * _p1 * mxy_u + _p2 * (rho2_u + 2.0 * mx2_u)
    distort_y = p[:, :, 1] + p[:, :, 1] * rad_dist_u + 2.0 * _p2 * mxy_u + _p1 * (rho2_u + 2.0 * my2_u)

    distort_x = distort_x * inputP[:, :, -1]
    distort_y = distort_y * inputP[:, :, -1]
    outputP = torch.cat([distort_x.unsqueeze(2), distort_y.unsqueeze(2), inputP[:, :, 2].unsqueeze(2)], dim=2)

    return outputP


def CameraToUV(inputP, K, Xi, cCmaD):  # pytorch version
    """
    Input:
        inputP: point in world corrdinate [bs,N,3]
        cCmaE: camera extrinsics [4,4]
        cCmaI: camera intrinsics [3,3]
        cCmaD: camera distortion coefficients [1,4]
    Return:
        outputUV: UV corrdinate [bs,N,2]
    """
    import torch.nn.functional as F

    # batch_size = inputP.shape[0]
    # 2. Normalize points in unit sphere
    inputP = F.normalize(inputP, p=2, dim=2)

    # 3. Transfer to a new corrdinate, which origin is (0,0,cCmaI[0][0])
    inputP[:, :, 2] += Xi

    # 4. Apply distortion
    inputP = distort_before_projection(inputP, cCmaD)

    # 5. Apply projection
    outputUV = inputP / inputP[:, :, -1].unsqueeze(-1)

    # # 4. Apply projection
    # outputUV = inputP / inputP[:,:,-1].unsqueeze(-1)

    # # 5. Apply distortion
    # outputUV = batch_distort(outputUV, cCmaD)

    # 6. Apply camera intrinsics
    # K = torch.zeros([batch_size, 3, 3], dtype=torch.float32, device=inputP.device)
    # K[:,0,0] = cCmaI[:,0,1]
    # K[:,1,1] = cCmaI[:,0,2]
    # K[:,2,2] = 1.
    # K[:,0,2] = cCmaI[:,1,0]
    # K[:,1,2] = cCmaI[:,1,1]
    K = torch.tensor(K, dtype=torch.float32).unsqueeze(0)
    outputUV = torch.einsum("bij,bkj->bki", K, outputUV)

    return outputUV[:, :, :-1]


class FisheyeCamera:
    def __init__(
        self,
        camera_epsilon,
        camera_intrinsics,
        camera_distortions,
        imu2camera_extrinsics,
        camera2imu_extrinsics,
    ):
        """
        Args:
            camera_epsilon (float): sclar
            camera_intrinsics (array): [3, 3]
            camera_distortions (array): [1, 4]
            imu2camera_extrinsics (array): [4, 4], SE3
            camera2imu_extrinsics (array): [4, 4], SE3
        """
        self.imu2camera_extrinsics = np.asarray(imu2camera_extrinsics)
        self.camera2imu_extrinsics = np.asarray(camera2imu_extrinsics)

        self.camera_epsilon = camera_epsilon
        self.camera_intrinsics = np.asarray(camera_intrinsics)
        self.camera_distortions = np.asarray(camera_distortions)

        self.fu_reciprocal = 1.0 / self.camera_intrinsics[0, 0]
        self.fv_reciprocal = 1.0 / self.camera_intrinsics[1, 1]
        self.cu = self.camera_intrinsics[0, 2]
        self.cv = self.camera_intrinsics[1, 2]

    def imu2camera(self, imu_points):
        """
        Args:
            imu_points (array): [3, N], xyz
        """
        return projection(imu_points, self.imu2camera_extrinsics)

    def camera2imu(self, camera_points):
        """
        Args:
            camera_points (array): [3, N], xyz
        """
        return projection(camera_points, self.camera2imu_extrinsics)

    def camera2pixel(self, camera_points):
        """
        Args:
            camera_points (array): [3, N], xyz
        """
        pixel_points = camera2pixel(
            camera_points,
            self.camera_epsilon,
            self.camera_distortions,
            self.camera_intrinsics,
        )
        return pixel_points

    def pixel2sphere(self, points_pixel):
        """
        Args:
            pixel_points (array): [N, 2], uv

        Returns:
            sphere_points (array): [N, 3], xyz
        """
        assert points_pixel.shape[1] == 2

        points_distort = np.asarray(
            (
                self.fu_reciprocal * (points_pixel[:, 0] - self.cu),
                self.fv_reciprocal * (points_pixel[:, 1] - self.cv),
            )
        )
        points_plane = distort2plane(points_distort, self.camera_distortions)
        points_sphere = plane2sphere(points_plane, self.camera_epsilon)
        return points_sphere


class OmniCamera(FisheyeCamera):
    def __init__(self, camera_params):
        """
        Args:
            camera_params (Iterable): length == 25
        """
        # camera_epsilon = camera_params[0]
        # camera_intrinsics = np.asarray(
        #     [
        #         [camera_params[1], 0.0, camera_params[3]],
        #         [0.0, camera_params[2], camera_params[4]],
        #         [0.0, 0.0, 1.0],
        #     ]
        # )
        # camera_distortions = np.asarray(camera_params[5:9])
        # imu2camera_extrinsics = np.asarray(camera_params[9:25]).reshape(4, 4)
        # camera2imu_extrinsics = np.linalg.inv(imu2camera_extrinsics)

        camera_epsilon = camera_params['Xi']
        camera_intrinsics  = camera_params['K']
        camera_distortions = camera_params['D']
        imu2camera_extrinsics = camera_params['E']  # imu2camera
        camera2imu_extrinsics = np.linalg.inv(imu2camera_extrinsics)

        super().__init__(
            camera_epsilon=camera_epsilon,
            camera_intrinsics=camera_intrinsics,
            camera_distortions=camera_distortions,
            imu2camera_extrinsics=imu2camera_extrinsics,
            camera2imu_extrinsics=camera2imu_extrinsics,
        )
