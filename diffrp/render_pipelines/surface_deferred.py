"""
Deferred surface render pipeline.
"""
import enum
import torch
import trimesh
import nvdiffrast.torch as dr
from typing import Optional, List
from dataclasses import dataclass, fields

from ..camera import Camera
from . import RasterizeContext
from ..shader_ops import gpu_f32, transform_point, transform_vector, float4, float3
from ..composite import alpha_blend, additive


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


@dataclass
class GBuffer:
    surf_out: SurfaceOutputStandard
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
    M: torch.Tensor
    
    # attributes
    color: Optional[torch.Tensor] = None
    uv: Optional[torch.Tensor] = None
    uv2: Optional[torch.Tensor] = None

    def normalize(self):
        if self.color is None:
            self.color = torch.ones_like(self.verts.xxxx)
        if self.uv is None:
            self.uv = torch.zeros_like(self.verts.xy)
        if self.uv2 is None:
            self.uv2 = torch.zeros_like(self.uv)
        return self


def cat_draw_call_data(rds: List[RenderData]):
    rds = [x.normalize() for x in rds]
    kw = dict()
    vs = []
    nors = []
    tris = []
    sts = [torch.full([1], 0, device='cuda')]
    offset = 0
    for s, x in enumerate(rds):
        vs.append(transform_point(x.verts, x.M).xyz)
        nors.append(transform_vector(x.normals, x.M))
        tris.append(x.tris + offset)
        sts.append(torch.full([len(x.tris)], s + 1, device=x.tris.device))
        offset += len(x.verts)
    stencil = torch.cat(sts)
    for field in fields(RenderData):
        if field.name not in ('verts', 'tris', 'normals', 'M'):
            kw[field.name] = torch.cat([getattr(x, field.name) for x in rds])
    I4 = torch.eye(4, device='cuda', dtype=torch.float32)
    return stencil, RenderData(
        verts=torch.cat(vs),
        tris=torch.cat(tris),
        normals=torch.cat(nors),
        M=I4, **kw
    )


@dataclass
class DrawCall:
    # material
    material: SurfaceMaterial
    render_data: RenderData


def stencil_match_g_buffer(g_buffers: List[GBuffer]):
    assert len(g_buffers) > 0
    allow_inplace = not torch.is_grad_enabled()
    ref_albedo = g_buffers[0].surf_out.albedo
    ref_stencil = g_buffers[0].stencil_match
    ref_rast = g_buffers[0].rast_inputs
    g = GBuffer(SurfaceOutputStandard(
        ref_albedo,
        torch.zeros_like(ref_albedo.xyz),
        torch.zeros_like(ref_albedo.xyz),
        torch.zeros_like(ref_albedo.x),
        torch.full_like(ref_albedo.x, 0.5),
        torch.ones_like(ref_albedo.x),
        torch.ones_like(ref_albedo.x)
    ), torch.zeros_like(ref_stencil), ref_rast)
    for n in g_buffers:
        for field in fields(SurfaceOutputStandard):
            cur = getattr(n.surf_out, field.name)
            if cur is None:
                continue
            pre = getattr(g.surf_out, field.name)
            setattr(g.surf_out, field.name, torch.where(n.stencil_match, cur, pre))
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
        if mode == SurfaceShading.Unlit:
            return g_buffer.surf_out.albedo
        if mode == SurfaceShading.Emission:
            return g_buffer.surf_out.emission
        if mode == SurfaceShading.FalseColorMask:
            return float3(
                g_buffer.surf_out.metallic,
                g_buffer.surf_out.smoothness,
                g_buffer.surf_out.occlusion
            )
        raise NotImplementedError("GBuffer Shader not implemented yet for the given shading mode", mode) 

    def record(self, dc: DrawCall):
        self.dcs.append(dc)
        return self

    def execute(
        self,
        ctx: RasterizeContext,
        opaque_only=True,
        shading: SurfaceShading = SurfaceShading.Unlit,
        g_buffers: List[GBuffer] = None
    ):
        if g_buffers is None:
            g_buffers = []
            stencil, render_data = cat_draw_call_data([x.render_data for x in self.dcs])
            v = render_data.verts
            tris = render_data.tris
            clip_space = transform_point(v, self.p @ self.v @ render_data.M)
            rng = torch.tensor([[0, len(tris)]], dtype=torch.int32)
            with dr.DepthPeeler(ctx, clip_space, tris, (self.h, self.w), rng) as dp:
                while True:
                    rast, rast_db = dp.rasterize_next_layer()
                    if (rast.a <= 0).all():
                        break
                    world_pos, _ = dr.interpolate(v, rast, tris, rast_db)
                    world_normal, _ = dr.interpolate(render_data.normals, rast, tris, rast_db)
                    view_dir = world_pos - self.camera_pos

                    color, _ = dr.interpolate(render_data.color, rast, tris, rast_db)
                    uv, uv_da = dr.interpolate(render_data.uv, rast, tris, rast_db, 'all')
                    uv2, uv2_da = dr.interpolate(render_data.uv2, rast, tris, rast_db, 'all')

                    surf_in = SurfaceInput(
                        view_dir, world_pos, world_normal,
                        color, uv, uv2, uv_da, uv2_da
                    )

                    stencil_b = stencil[rast.a.long()]
                    layer_gbuf = []
                    for s, dc in enumerate(self.dcs):
                        surf_uni = SurfaceUniform(dc.render_data.M, self.v, self.p)
                        surf_out = dc.material.shade(surf_uni, surf_in)
                        stencil_match = stencil_b == s + 1
                        layer_gbuf.append(GBuffer(surf_out, stencil_match, (rast, clip_space, tris)))
                    g_buffers.append(stencil_match_g_buffer(layer_gbuf))
                    # opaque needs one layer
                    if opaque_only:
                        break
        for g in reversed(g_buffers):
            rgb = self.shade_g_buffer(g, shading)
            alpha = g.surf_out.alpha
            rgba = float4(rgb, alpha)
            # if antialias:
            #     rgba = dr.antialias(rgba, *g.rast_inputs)
            if shading == SurfaceShading.LitAdditive:
                blend_fn = additive
            else:
                blend_fn = alpha_blend
            self.frame_b = torch.where(g.stencil_match, blend_fn(self.frame_b, rgba), self.frame_b)
        return g_buffers, self.frame_b.squeeze(0)
