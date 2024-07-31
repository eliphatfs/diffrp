"""
Deferred surface render pipeline.
"""
import enum
import torch
import trimesh
import nvdiffrast.torch as dr
from typing import Optional, List
from dataclasses import dataclass, fields

from ..utils.shader_ops import *
from .camera import Camera
from . import RasterizeContext
from ..utils.composite import alpha_blend, additive


@dataclass
class SurfaceUniform:
    M: torch.Tensor
    V: torch.Tensor
    P: torch.Tensor


@dataclass
class SurfaceInput:
    view_dir: torch.Tensor  # F3, view direction (not normalized)
    world_pos: torch.Tensor  # F3, world position
    world_normal: torch.Tensor  # F3, geometry world normal (not normalized)

    color: torch.Tensor  # F4, vertex color, default to ones
    uv: torch.Tensor  # F2, uv, default to zeros
    uv2: torch.Tensor  # F2, optional extra set of uv, default to zeros

    uv_da: torch.Tensor  # derivatives for uv, default to zeros
    uv2_da: torch.Tensor  # derivatives for uv2, default to zeros


@dataclass
class SurfaceOutputStandard:
    albedo: torch.Tensor  # F3, base color (diffuse or specular)
    normal: Optional[torch.Tensor] = None  # F3, tangent space normal
    emission: Optional[torch.Tensor] = None  # F3, default to black
    metallic: Optional[torch.Tensor] = None  # F1, default to 0.0
    smoothness: Optional[torch.Tensor] = None  # F1, default to 0.5
    occlusion: Optional[torch.Tensor] = None  # F1, default to 1.0
    alpha: Optional[torch.Tensor] = None  # F1, default to 1.0

    def to_gbuffer_tensor(self, tangents=None):
        albedo = self.albedo
        return torch.cat([
            albedo,
            zeros_like_vec(albedo, 3) + albedo.new_tensor([0, 0, 1]) if self.normal is None else self.normal,
            zeros_like_vec(albedo, 3) if self.emission is None else self.emission,
            zeros_like_vec(albedo, 1) if self.metallic is None else self.metallic,
            full_like_vec(albedo, 0.5, 1) if self.smoothness is None else self.smoothness,
            ones_like_vec(albedo, 1) if self.occlusion is None else self.occlusion,
            ones_like_vec(albedo, 1) if self.alpha is None else self.alpha,
            zeros_like_vec(albedo, 4) if tangents is None else tangents
        ], dim=-1)


@dataclass
class GTensorView(SurfaceOutputStandard):
    tangents: Optional[torch.Tensor] = None

    @staticmethod
    def from_gbuffer_tensor(g_tensor: torch.Tensor):
        return GTensorView(
            g_tensor[..., 0:3],
            g_tensor[..., 3:6],
            g_tensor[..., 6:9],
            g_tensor[..., 9:10],
            g_tensor[..., 10:11],
            g_tensor[..., 11:12],
            g_tensor[..., 12:13],
            g_tensor[..., 13:17]
        )

@dataclass
class GBuffer:
    g_tensor: torch.Tensor
    stencil_match: torch.BoolTensor
    rast_inputs: tuple


class SurfaceShading(enum.Enum):
    Unlit = 1  # unlit, or albedo only
    Emission = 2  # emission only
    LitCutout = 3  # alpha clipping with threshold 0.5
    LitFade = 4  # alpha dims everything, alpha blending
    LitAdditive = 5  # alpha dims everything, additive blending
    LitPhysical = 6  # alpha dims diffuse but keeps specular
    FalseColorNormal = 7  # camera space normal
    FalseColorMask = 8  # R = metal G = smooth B = occlusion


class SurfaceMaterial:

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        raise NotImplementedError


@dataclass
class RenderData:
    # geometry
    verts: torch.Tensor
    tris: torch.IntTensor
    normals: torch.Tensor

    # transform
    M: Optional[torch.Tensor] = None
    
    # attributes
    color: Optional[torch.Tensor] = None
    uv: Optional[torch.Tensor] = None
    uv2: Optional[torch.Tensor] = None
    tangents: Optional[torch.Tensor] = None

    def normalize(self):
        if self.color is None:
            self.color = ones_like_vec(self.verts, 4)
        if self.uv is None:
            self.uv = zeros_like_vec(self.verts, 2)
        if self.uv2 is None:
            self.uv2 = torch.zeros_like(self.uv)
        if self.tangents is None:
            self.tangents = torch.zeros_like(self.color)
        return self


def cat_draw_call_data(rds: List[RenderData]):
    rds = [x.normalize() for x in rds]
    kw = dict()
    vs = []
    nors = []
    tans = []
    tris = []
    sts = [torch.full([1], 0, device='cuda', dtype=torch.int16)]
    offset = 0
    for s, x in enumerate(rds):
        if x.M is None:
            vs.append(x.verts)
            nors.append(x.normals)
            tans.append(x.tangents)
        else:
            vs.append(transform_point4x3(x.verts, x.M))
            nors.append(transform_vector(x.normals, x.M))
            tans.append(float4(transform_vector(x.tangents[..., :3], x.M), x.tangents[..., 3:]))
        tris.append(x.tris + offset)
        sts.append(x.tris.new_full([len(x.tris)], s + 1, dtype=torch.int16))
        offset += len(x.verts)
    stencil = torch.cat(sts)
    for field in fields(RenderData):
        if field.name not in ('verts', 'tris', 'normals', 'M', 'tangents'):
            kw[field.name] = torch.cat([getattr(x, field.name) for x in rds])
    return stencil, RenderData(
        verts=torch.cat(vs),
        tris=torch.cat(tris),
        normals=torch.cat(nors),
        M=None,
        tangents=torch.cat(tans),
        **kw
    )


@dataclass
class DrawCall:
    # material
    material: SurfaceMaterial
    render_data: RenderData


def stencil_match_g_buffer(g_buffers: List[GBuffer]):
    assert len(g_buffers) > 0
    allow_inplace = not torch.is_grad_enabled()
    g = g_buffers[0]
    for n in g_buffers[1:]:
        g.g_tensor = torch.where(n.stencil_match, n.g_tensor, g.g_tensor)
        if allow_inplace:
            g.stencil_match |= n.stencil_match
        else:
            g.stencil_match = g.stencil_match | n.stencil_match
    return g


class SurfaceDeferredRenderPipeline:
    def __init__(self) -> None:
        self.dcs: List[DrawCall] = []

    def new_frame(self, camera: Camera, bg_color: List[float]):
        self.dcs.clear()
        self.camera_pos = gpu_f32(trimesh.transformations.translation_from_matrix(camera.t))
        self.v = gpu_f32(camera.V())
        self.p = gpu_f32(camera.P())
        self.h, self.w = camera.resolution()
        self.frame_b = torch.empty([1, self.h, self.w, 4], device='cuda', dtype=torch.float32)
        self.frame_b[:] = gpu_f32(bg_color)
        return self

    def shade_g_buffer(self, g_buffer: GBuffer, mode: SurfaceShading):
        surf_in: SurfaceInput = g_buffer.rast_inputs[-1]
        surf_out = GTensorView.from_gbuffer_tensor(g_buffer.g_tensor)
        if mode == SurfaceShading.Unlit:
            return surf_out.albedo
        if mode == SurfaceShading.Emission:
            return surf_out.emission
        if mode == SurfaceShading.FalseColorMask:
            return float3(
                surf_out.metallic,
                surf_out.smoothness,
                surf_out.occlusion
            )
        if mode == SurfaceShading.FalseColorNormal:
            vnt = surf_out.normal
            vn = surf_in.world_normal
            vt, vs = split_alpha(surf_out.tangents)
            vb = vs * torch.cross(vn, vt, dim=-1)
            return normalized(transform_vector(vnt.x * vt + vnt.y * vb + vnt.z * vn, self.v)) * 0.5 + 0.5
        raise NotImplementedError("GBuffer Shader not implemented yet for the given shading mode", mode) 

    def record(self, dc: DrawCall):
        self.dcs.append(dc)
        return self

    def execute(
        self,
        ctx: RasterizeContext,
        opaque_only=True,
        shading: SurfaceShading = SurfaceShading.Unlit,
        g_buffers: Optional[List[GBuffer]] = None
    ):
        if g_buffers is None:
            g_buffers = []
            stencil, render_data = cat_draw_call_data([x.render_data for x in self.dcs])
            v = render_data.verts
            tris = render_data.tris
            assert render_data.M is None
            clip_space = transform_point(v, self.p @ self.v)
            rng = torch.tensor([[0, len(tris)]], dtype=torch.int32)
            with dr.DepthPeeler(ctx, clip_space, tris, (self.h, self.w), rng) as dp:
                while True:
                    rast, rast_db = dp.rasterize_next_layer()
                    if (not opaque_only) and (rast[..., -1] <= 0).all():
                        break
                    world_pos, _ = dr.interpolate(v, rast, tris, rast_db)
                    world_normal, _ = dr.interpolate(render_data.normals, rast, tris, rast_db)
                    view_dir = world_pos - self.camera_pos

                    color, _ = dr.interpolate(render_data.color, rast, tris, rast_db)
                    uv, uv_da = dr.interpolate(render_data.uv, rast, tris, rast_db, 'all')
                    uv2, uv2_da = dr.interpolate(render_data.uv2, rast, tris, rast_db, 'all')
                    tangents, _ = dr.interpolate(render_data.tangents, rast, tris, rast_db)

                    surf_in = SurfaceInput(
                        view_dir, world_pos, world_normal,
                        color, uv, uv2, uv_da, uv2_da
                    )

                    stencil_b = stencil[rast.a.int()]
                    layer_gbuf = []
                    for s, dc in enumerate(self.dcs):
                        surf_uni = SurfaceUniform(dc.render_data.M, self.v, self.p)
                        surf_out = dc.material.shade(surf_uni, surf_in)
                        g = surf_out.to_gbuffer_tensor(tangents)
                        stencil_match = stencil_b == s + 1
                        layer_gbuf.append(GBuffer(g, stencil_match, (rast, clip_space, tris, surf_in)))
                    g_buffers.append(stencil_match_g_buffer(layer_gbuf))
                    # opaque needs one layer
                    if opaque_only:
                        break
        for g in reversed(g_buffers):
            rgb = self.shade_g_buffer(g, shading)
            alpha = GTensorView.from_gbuffer_tensor(g.g_tensor).alpha if not opaque_only else 1.0
            rgba = float4(rgb, alpha)
            # if antialias:
            #     rgba = dr.antialias(rgba, *g.rast_inputs)
            if shading == SurfaceShading.LitAdditive:
                blend_fn = additive
            else:
                blend_fn = alpha_blend
            self.frame_b = torch.where(g.stencil_match, blend_fn(self.frame_b, rgba), self.frame_b)
        return g_buffers, torch.flipud(self.frame_b.squeeze(0))
