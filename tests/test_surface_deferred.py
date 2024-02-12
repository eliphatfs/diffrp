import time
import numpy
import torch
import trimesh
import unittest
import trimesh.creation
import nvdiffrast.torch as dr
import matplotlib.pyplot as plotlib
from diffrp.camera import PerspectiveCamera
from diffrp.render_pipelines.surface_deferred import *
from diffrp.shader_ops import *


class ObjectSpaceVertexNormalMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(
            normalized(transform_vector(si.world_normal, torch.linalg.inv(su.M))) / 2 + 0.5
        )


class CameraSpaceVertexNormalMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(
            transform_vector(normalized(si.world_normal), su.V) / 2 + 0.5
        )


def make_face_soup(verts, tris, face_normals):
    return (
        verts[tris].reshape(-1, 3),
        numpy.arange(len(tris.reshape(-1))).reshape(tris.shape),
        numpy.stack([face_normals] * 3, axis=1).reshape(-1, 3)
    )


class TestSurfaceDeferredRP(unittest.TestCase):

    @torch.no_grad()
    def test_cylinder(self):
        # renders a cylinder
        rp = SurfaceDeferredRenderPipeline()
        cam = PerspectiveCamera(h=512, w=512)
        mesh = trimesh.creation.cylinder(0.3, 1.0)
        v, f, vn = make_face_soup(mesh.vertices, mesh.faces, mesh.face_normals)
        ctx = dr.RasterizeCudaContext()
        t = time.perf_counter()
        for i in range(500):
            M = gpu_f32(trimesh.transformations.identity_matrix()[[0, 2, 1, 3]])
            rp.new_frame(cam, [0.8, 0.8, 1.0, 0.6])
            rp.record(DrawCall(
                CameraSpaceVertexNormalMaterial(),
                RenderData(gpu_f32(v), gpu_i32(f), gpu_f32(vn), M)
            ))
            fb = rp.execute(ctx)[1]
        fb = fb.cpu().numpy()
        print(500 / (time.perf_counter() - t))
        plotlib.imshow(fb)
        plotlib.show()
