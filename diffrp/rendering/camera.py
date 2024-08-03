import math
import numpy
import trimesh
import calibur
from typing import Union, List
from ..utils.shader_ops import gpu_f32


class Camera:

    def __init__(self) -> None:
        self.t = trimesh.transformations.translation_matrix([0, 0.0, 3.2])

    def V(self):
        return gpu_f32(trimesh.transformations.inverse_matrix(self.t))

    def P(self):
        raise NotImplementedError

    def resolution(self):
        """
        returns: h, w
        """
        raise NotImplementedError


class PerspectiveCamera(Camera):
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
    def from_orbit(cls, h, w, radius, azim, elev, origin):
        r = radius
        cam = cls(h=h, w=w)
        theta, phi = math.radians(azim), math.radians(elev)
        z = r * math.cos(theta) * math.cos(phi)
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(phi)
        cam.t = trimesh.transformations.translation_matrix([x, y, z])
        cam.lookat(origin)
        return cam
