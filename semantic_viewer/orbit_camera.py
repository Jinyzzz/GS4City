#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import numpy as np
from scipy.spatial.transform import Rotation as R


class OrbitCamera:
    def __init__(self, W, H, r=2.0, fovy=60.0):
        self.W = W
        self.H = H
        self.radius = r
        self.center = np.array([0, 0, 0], dtype=np.float32)

        self.rot = R.from_quat([0, 0, 0, 1])

        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0

        self.rot_mode = 1

    @property
    def pose_movecenter(self):
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        res[:3, 3] -= self.center

        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]

        return res

    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        res[2, 3] += self.radius
        res[:3, 3] -= self.center

        res[:3, :3] = rot[:3, :3].transpose()

        return res

    @property
    def pose(self):
        if self.rot_mode == 1:
            return self.pose_movecenter
        else:
            return self.pose_objcenter

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        else:
            up = -self.up
            side = -self.right

        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.2 ** (-delta)
        self.radius = float(np.clip(self.radius, 0.05, 1e3))

    def pan(self, dx, dy, dz=0.0):
        if self.rot_mode == 1:
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        else:
            self.center += 0.0005 * np.array([-dx, dy, dz])
