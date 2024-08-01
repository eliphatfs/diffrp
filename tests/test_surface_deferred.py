import os
import glob
import numpy
import torch
import trimesh
import unittest
import trimesh.creation
import nvdiffrast.torch as dr
import matplotlib.pyplot as plotlib
from diffrp.rendering.camera import PerspectiveCamera
from diffrp.materials.base_material import SurfaceInput, SurfaceMaterial, SurfaceUniform, SurfaceOutputStandard
from diffrp.rendering.surface_deferred import SurfaceDeferredRenderSession
from diffrp.scene import Scene, MeshObject
from diffrp.utils.shader_ops import *
from diffrp.loaders.gltf_loader import load_gltf_scene
from diffrp.utils.colors import linear_to_srgb


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


class TestSurfaceDeferredRP(unittest.TestCase):

    @torch.no_grad()
    def test_cylinder(self):
        # renders a cylinder
        cam = PerspectiveCamera(h=512, w=512)
        mesh = trimesh.creation.cylinder(0.3, 1.0)
        v, f = mesh.vertices, mesh.faces
        # v, f, vn = make_face_soup(mesh.vertices, mesh.faces, mesh.face_normals)
        ctx = dr.RasterizeCudaContext()
        # t = time.perf_counter()
        scene = Scene()
        scene.add_mesh_object(MeshObject(
            CameraSpaceVertexNormalMaterial(),
            gpu_f32(v), gpu_i32(f),
            M=gpu_f32(trimesh.transformations.identity_matrix()[[0, 2, 1, 3]])
        ))
        for _ in range(5):
            rp = SurfaceDeferredRenderSession(ctx, scene, cam, False)
            fb = rp.false_color_camera_space_normal()
        fb = fb.cpu().numpy()
        plotlib.imsave("tmp/test/cylinder-transparent.png", fb)
        # print(500 / (time.perf_counter() - t))
        # plotlib.imshow(fb)
        # plotlib.show()


@torch.no_grad()
def normalize(gltf: Scene):
    bmin = gpu_f32([1e30] * 3)
    bmax = gpu_f32([-1e30] * 3)
    world_v = [transform_point4x3(prim.verts, prim.M) for prim in gltf.objects]
    for verts, prim in zip(world_v, gltf.objects):
        bmin = torch.minimum(bmin, verts.min(0)[0])
        bmax = torch.maximum(bmax, verts.max(0)[0])
    center = (bmin + bmax) / 2
    radius = max(length(verts - center).max() for verts in world_v).item()
    T = trimesh.transformations.translation_matrix(-center.cpu().numpy())
    S = trimesh.transformations.scale_matrix(1 / radius)
    M = gpu_f32(S @ T)
    for prim in gltf.objects:
        prim.M = M @ prim.M
    return gltf


class TestGLTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ctx = dr.RasterizeCudaContext()

    def render_gltf(self, fp):
        name = os.path.splitext(os.path.basename(fp))[0]
        ctx = self.ctx
        cam = PerspectiveCamera.from_orbit(640, 640, 3.8, 30, 20, [0, 0, 0])
        gltf = normalize(load_gltf_scene(fp, compute_tangents=True))
        rp = SurfaceDeferredRenderSession(ctx, gltf, cam, False)
        frame = rp.albedo()
        frame = float4(linear_to_srgb(frame.rgb), frame.a)
        plotlib.imsave("tmp/test/%s-albedo.png" % name, saturate(frame).cpu().numpy())
        frame = rp.false_color_mask_mso()
        plotlib.imsave("tmp/test/%s-mask.png" % name, saturate(frame).cpu().numpy())
        frame = rp.false_color_camera_space_normal()
        plotlib.imsave("tmp/test/%s-normal.png" % name, saturate(frame).cpu().numpy())

    @torch.no_grad()
    def test_gltfs(self):
        for fp in glob.glob("tmp/data/*.glb"):
            self.render_gltf(fp)
