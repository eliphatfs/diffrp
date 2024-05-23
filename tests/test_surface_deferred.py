import os
import glob
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
from diffrp.geometry import compute_vertex_normals
from diffrp.loaders.gltf_loader import GLTFLoader
from diffrp.colors import linear_to_srgb


class ObjectSpaceVertexNormalMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(
            normalized(transform_vector(si.world_normal, torch.linalg.inv(su.M))) / 2 + 0.5
        )


class CameraSpaceVertexNormalMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(
            transform_vector(normalized(si.world_normal), su.V) / 2 + 0.5,
            alpha=full_like_vec(si.world_normal, 0.5, 1)
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
        v, f = mesh.vertices, mesh.faces
        # v, f, vn = make_face_soup(mesh.vertices, mesh.faces, mesh.face_normals)
        ctx = dr.RasterizeCudaContext()
        # t = time.perf_counter()
        for _ in range(5):
            M = gpu_f32(trimesh.transformations.identity_matrix()[[0, 2, 1, 3]])
            rp.new_frame(cam, [0.8, 0.8, 1.0, 0.1])
            rp.record(DrawCall(
                CameraSpaceVertexNormalMaterial(),
                RenderData(gpu_f32(v), gpu_i32(f), compute_vertex_normals(gpu_f32(v), gpu_i32(f)), M)
            ))
            fb = rp.execute(ctx, opaque_only=False)[1]
        fb = fb.cpu().numpy()
        plotlib.imsave("tmp/test/cylinder-transparent.png", fb)
        # print(500 / (time.perf_counter() - t))
        # plotlib.imshow(fb)
        # plotlib.show()


@torch.no_grad()
def normalize(gltf: GLTFLoader):
    bmin = gpu_f32([1e30] * 3)
    bmax = gpu_f32([-1e30] * 3)
    world_v = [transform_point4x3(prim.verts, prim.transform) for prim in gltf.prims]
    for verts, prim in zip(world_v, gltf.prims):
        bmin = torch.minimum(bmin, verts.min(0)[0])
        bmax = torch.maximum(bmax, verts.max(0)[0])
    center = (bmin + bmax) / 2
    radius = max(length(verts - center).max() for verts in world_v).item()
    T = trimesh.transformations.translation_matrix(-center.cpu().numpy())
    S = trimesh.transformations.scale_matrix(1 / radius)
    M = gpu_f32(S @ T)
    for prim in gltf.prims:
        prim.transform = M @ prim.transform
    return gltf


class TestGLTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ctx = dr.RasterizeCudaContext()

    def render_gltf(self, fp):
        name = os.path.splitext(os.path.basename(fp))[0]
        ctx = self.ctx
        rp = SurfaceDeferredRenderPipeline()
        cam = PerspectiveCamera.from_orbit(640, 640, 3.8, 30, 70, [0, 0, 0])
        rp.new_frame(cam, [1.0, 1.0, 1.0, 0.0])
        gltf = normalize(GLTFLoader(fp, compute_tangents=True))
        for prim in gltf.prims:
            rp.record(DrawCall(prim.material, RenderData(
                prim.verts, prim.tris, prim.normals,
                M=prim.transform,
                color=prim.color, uv=prim.uv,
                tangents=prim.tangents
            )))
        g, frame = rp.execute(ctx, shading=SurfaceShading.Unlit, opaque_only=False)
        frame = float4(linear_to_srgb(frame.rgb), frame.a)
        plotlib.imsave("tmp/test/%s-albedo.png" % name, saturate(frame).cpu().numpy())
        _, frame = rp.execute(ctx, shading=SurfaceShading.FalseColorMask, g_buffers=g, opaque_only=False)
        plotlib.imsave("tmp/test/%s-mask.png" % name, saturate(frame).cpu().numpy())
        _, frame = rp.execute(ctx, shading=SurfaceShading.FalseColorNormal, g_buffers=g, opaque_only=False)
        plotlib.imsave("tmp/test/%s-normal.png" % name, saturate(frame).cpu().numpy())

    @torch.no_grad()
    def test_gltfs(self):
        for fp in glob.glob("tmp/data/*.glb"):
            self.render_gltf(fp)
