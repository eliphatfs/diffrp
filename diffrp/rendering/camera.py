import math
import numpy
import torch
import trimesh
import calibur
from typing import Union, List
from ..utils.shader_ops import gpu_f32


class Camera:
    """
    Abstract class for camera specification.

    You need to implement the ``V()``, ``P()`` and ``resolution()`` methods
    if you inherit from the class.
    """

    def __init__(self) -> None:
        self.t = trimesh.transformations.translation_matrix([0, 0.0, 3.2])

    def V(self):
        """
        Returns:
            torch.Tensor:
                | GPU tensor of shape (4, 4). GL view matrix.
                | Inverse of the c2w or camera pose transform matrix.
                | You may need to convert the camera poses if the pose is not in GL convention
                  (X right, Y up, -Z forward). You may use ``calibur`` for this.
                | It should be an affine transform matrix with the last row fixed to [0, 0, 0, 1].
                  It is also often called an extrinsic matrix or w2c matrix in other conventions.
        """
        return gpu_f32(trimesh.transformations.inverse_matrix(self.t).astype(numpy.float32))

    def P(self):
        """
        Returns:
            torch.Tensor: GPU tensor of shape (4, 4). GL projection matrix.
        """
        raise NotImplementedError

    def resolution(self):
        """
        Returns:
            Tuple[int, int]: Resolution in (h, w) order.
        """
        raise NotImplementedError


class RawCamera(Camera):
    """
    Raw data-driven camera that takes GPU tensors of the view and projection matrices to drive the camera.
    """
    def __init__(self, h: int, w: int, v: torch.Tensor, p: torch.Tensor) -> None:
        self.h = h
        self.w = w
        self.v = v
        self.p = p

    def V(self):
        return self.v

    def P(self):
        return self.p

    def resolution(self):
        return self.h, self.w

class PerspectiveCamera(Camera):
    """
    Perspective camera class. Angles are in degrees.
    """
    def __init__(self, fov=30, h=512, w=512, near=0.1, far=10.0) -> None:
        super().__init__()
        self.fov = fov
        self.h = h
        self.w = w
        self.near = near
        self.far = far

    def P(self):
        cx, cy = self.w / 2, self.h / 2
        fx = fy = calibur.fov_to_focal(numpy.radians(self.fov), self.h)
        return gpu_f32(calibur.projection_gl_persp(
            self.w, self.h,
            cx, cy,
            fx, fy,
            self.near, self.far
        ))

    def resolution(self):
        return self.h, self.w

    def set_transform(self, tr: numpy.ndarray):
        self.t = tr

    def lookat(self, point: Union[List[int], numpy.ndarray]):
        if isinstance(point, list):
            point = numpy.array(point)
        assert list(point.shape) == [3]
        pos = trimesh.transformations.translation_from_matrix(self.t)
        fwd = point - pos
        z = -fwd
        x = numpy.cross(fwd, [0, 1, 0])
        y = numpy.cross(x, fwd)
        self.t[:3, 0] = calibur.normalized(x)
        self.t[:3, 1] = calibur.normalized(y)
        self.t[:3, 2] = calibur.normalized(z)
        return self

    @classmethod
    def from_orbit(cls, h, w, radius, azim, elev, origin, fov=30, near=0.1, far=10.0):
        """
        Create a perspective camera from an orbital camera.

        Args:
            h (int): Height.
            w (int): Width.
            radius (float): Distance to focus origin.
            azim (float): Azimuth angle in degrees relative to focus origin. +Z is 0 degree.
            elev (float): Elevation angle in degrees relative to focus origin.
                -90 is down-up and 90 is up-down.
                Note that -90 and 90 are extreme values where azimuth is undefined,
                so use values like 89.999 in these cases instead.
            origin (List[float]): List of 3 floats, focus point of the camera.
            fov (float): Vertical Field of View in degrees.
            near (float): Camera near plane distance.
            far (float): Camera far plane distance.
        """
        r = radius
        cam = cls(h=h, w=w, fov=fov, near=near, far=far)
        theta, phi = math.radians(azim), math.radians(elev)
        z = r * math.cos(theta) * math.cos(phi) + origin[2]
        x = r * math.sin(theta) * math.cos(phi) + origin[0]
        y = r * math.sin(phi) + origin[1]
        cam.t = trimesh.transformations.translation_matrix([x, y, z])
        cam.lookat(origin)
        return cam
