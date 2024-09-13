"""
Experimental. Implementation and interface subject to change.
"""
import math
import torch
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Optional, Dict, Callable

from .camera import Camera
from ..utils.cache import cached
from ..utils.shader_ops import *
from .mixin import RenderSessionMixin
from ..utils.geometry import barycentric
from ..scene import Scene, ImageEnvironmentLight
from ..utils.raycaster import NaivePBBVH, BruteForceRaycaster, TorchOptiX
from .interpolator import MaskedSparseInterpolator, FullScreenInterpolator
from ..utils.coordinates import near_plane_ndc_grid, unit_direction_to_latlong_uv
from ..materials.base_material import SurfaceInput, SurfaceUniform, SurfaceOutputStandard
from ..utils.light_transport import hammersley, importance_sample_ggx, combine_fixed_tangent_space, geometry_smith, fresnel_schlick_smoothness


@dataclass
class PathTracingSessionOptions:
    ray_depth: int = 3
    ray_spp: int = 16
    ray_split_size: int = 8 * 1024 * 1024

    deterministic: bool = True
    
    pbr_ray_step_epsilon: float = 1e-3
    pbr_ray_last_bounce: Literal['void', 'skybox'] = 'void'

    raycaster_impl: Literal['brute-force', 'naive-pbbvh', 'torchoptix'] = 'torchoptix'
    raycaster_epsilon: float = 1e-8
    raycaster_builder: Literal['morton', 'splitaxis'] = 'splitaxis'


@dataclass
class RayOutputs:
    radiance: torch.Tensor
    transfer: torch.Tensor
    next_rays_o: torch.Tensor
    next_rays_d: torch.Tensor
    alpha: torch.Tensor
    extras: Dict[str, torch.Tensor]


class PathTracingSession(RenderSessionMixin):
    """
    Experimental path tracing pipeline. Implementation and interface subject to change.
    """
    def __init__(
        self,
        scene: Scene, camera: Camera,
        options: Optional[PathTracingSessionOptions] = None
    ) -> None:
        self.scene = scene
        self.camera = camera
        if options is None:
            options = PathTracingSessionOptions()
        self.options = options

    @cached
    def raycaster(self):
        vao = self.vertex_array_object()
        cfg = {'epsilon': self.options.raycaster_epsilon}
        if self.options.raycaster_impl == 'torchoptix':
            return TorchOptiX(vao.world_pos, vao.tris, cfg)
        elif self.options.raycaster_impl == 'naive-pbbvh':
            if len(vao.tris) <= 3:
                return BruteForceRaycaster(vao.world_pos, vao.tris, cfg)
            else:
                cfg['builder'] = self.options.raycaster_builder
                return NaivePBBVH(vao.world_pos, vao.tris, cfg)
        else:
            return BruteForceRaycaster(vao.world_pos, vao.tris, cfg)

    def layer_material_rays(self, rays_o, rays_d, t, i: torch.Tensor):
        cam_v = self.camera_V()
        cam_p = self.camera_P()
        far = self.camera_far()
        vao = self.vertex_array_object()
        i = torch.where(t < far, i.float() + 1, 0)
        vi_data = float4(
            barycentric(*torch.unbind(vao.world_pos[vao.tris[(i - 1).long()]], dim=-2), rays_o + rays_d * t[..., None]).xy,
            t[..., None], i[..., None]
        ).view(-1, 4)
        mats = []
        stencil_buf = vao.stencils[i.long()]
        for pi, x in enumerate(self.scene.objects):
            su = SurfaceUniform(x.M, cam_v, cam_p)
            si = SurfaceInput(su, vao, MaskedSparseInterpolator(vi_data, vao.tris, stencil_buf == pi + 1))
            if len(si.interpolator.indices[0]) == 0:
                mats.append((si, SurfaceOutputStandard()))
                continue
            so = x.material.shade(su, si)
            mats.append((si, so))
        return mats

    def _super_collector(self, si: SurfaceInput, so: SurfaceOutputStandard):
        albedo = so.albedo if so.albedo is not None else gpu_f32([1.0, 0.0, 1.0]).expand_as(world_normal)
        world_normal = self._collector_world_normal(si, so)
        metal = so.metallic if so.metallic is not None else zeros_like_vec(world_normal, 1)
        smooth = so.smoothness if so.smoothness is not None else full_like_vec(world_normal, 0.5, 1)
        alpha = so.alpha if so.alpha is not None else full_like_vec(world_normal, 1, 1)
        emission = so.emission if so.emission is not None else full_like_vec(world_normal, 0, 3)
        return torch.cat([albedo, world_normal, metal, smooth, alpha, emission], dim=-1)
    
    @staticmethod
    @torch.jit.script
    def _sampler_brdf_impl(attrs: torch.Tensor, t: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor, env_radiance: torch.Tensor):
        albedo = attrs[..., 0:3]
        world_normal = attrs[..., 3:6]
        metal = attrs[..., 6:7]
        smooth = attrs[..., 7:8]
        alpha = attrs[..., 8:9]
        emission = attrs[..., 9:12]
        next_o = rays_o + rays_d * t[..., None]

        dielectric = 1 - metal
        diffuse_color = dielectric * albedo
        d = diffuse_color.max(-1, True).values
        diffuse_prob = dielectric * d / (0.04 + d)
        specular_prob = 1 - diffuse_prob  # 0.04 * dielectric + metal
        ray_is_transmit = torch.rand_like(alpha) >= alpha
        ray_is_diffuse = torch.rand_like(metal) >= specular_prob
        
        # transmit
        next_d_tr = rays_d
        transfer_tr = 1.0
        
        # diffuse
        z2 = torch.rand_like(metal)
        theta = torch.rand_like(z2) * math.tau
        xy = torch.sqrt(1.0 - z2)
        next_d_di = combine_fixed_tangent_space(xy * torch.cos(theta), z2.sqrt(), xy * torch.sin(theta), world_normal)
        diffuse_color = albedo * dielectric
        transfer_di = diffuse_color / torch.clamp_min(diffuse_prob, 0.0001)
        
        # specular
        roughness_clip = (1 - smooth).clamp_min(1 / 512)
        x = torch.rand_like(smooth)
        y = torch.rand_like(smooth)
        h = importance_sample_ggx(x, y, world_normal, roughness_clip)
        next_d_sp = reflect(rays_d, h)
        dot_v_h = -dot(h, rays_d)
        f0 = fma(albedo, metal, 0.04 * dielectric)
        transfer_sp = (
            fresnel_schlick_smoothness(dot_v_h, f0, smooth) *
            geometry_smith(world_normal, -rays_d, next_d_sp, roughness_clip) *
            torch.clamp_min(dot_v_h, 1e-6) / (torch.clamp_min(dot(world_normal, h), 1e-6) * torch.clamp_min(-dot(world_normal, rays_d), 1e-6))
        ) / torch.clamp_min(specular_prob * alpha, 0.0001)
        
        next_d = torch.where(ray_is_transmit, next_d_tr, torch.where(ray_is_diffuse, next_d_di, next_d_sp))
        transfer = torch.where(ray_is_transmit, transfer_tr, torch.where(ray_is_diffuse, transfer_di, transfer_sp))
        return albedo, emission, world_normal, alpha, emission + env_radiance, transfer, next_o, next_d

    @cached
    def _single_env_light(self):
        env_light = None
        for light in self.scene.lights:
            if isinstance(light, ImageEnvironmentLight):
                if env_light is not None:
                    raise ValueError("Only one environment light is supported in path tracing now.")
                env_light = light.image_rh()
        if env_light is None:
            return black_tex().rgb
        return env_light

    def sampler_brdf(self, rays_o, rays_d, t, i, d: int):
        far = self.camera_far()
        mats = self.layer_material_rays(rays_o, rays_d, t, i)
        always_sky = self.options.ray_depth - 1 == d and self.options.pbr_ray_last_bounce == 'skybox'
        attrs = self._gbuffer_collect_layer_impl_stencil_masked(
            mats, self._super_collector, zeros_like_vec(rays_o, 12)
        )
        
        hit = t[..., None] < far
        env_radiance = sample2d(self._single_env_light(), float2(*unit_direction_to_latlong_uv(rays_d)))
        if not always_sky:
            env_radiance = torch.where(hit, 0.0, env_radiance)
        albedo, emission, world_normal, alpha, radiance, transfer, next_o, next_d = PathTracingSession._sampler_brdf_impl(attrs, t, rays_o, rays_d, env_radiance)

        return RayOutputs(
            alpha=alpha,
            radiance=radiance,
            transfer=torch.where(hit, transfer, 0.0),
            next_rays_o=next_o + next_d * self.options.pbr_ray_step_epsilon,
            next_rays_d=next_d,
            extras=dict(albedo=albedo, emission=emission, world_normal=world_normal, world_position=next_o)
        )

    def trace_rays(self, sampler: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], RayOutputs], radiance_channels: int = 3):
        device = torch.device('cuda')
        raycaster = self.raycaster()
        far = self.camera_far()
        cam_v = self.camera_V()
        cam_p = self.camera_P()
        H, W = self.camera.resolution()
        pix_grid = near_plane_ndc_grid(H, W, torch.float32, device).reshape(-1, 4)
        qrx, qry = hammersley(self.options.ray_spp, self.options.deterministic, device)
        sections = min(self.options.ray_spp, math.ceil((H * W * self.options.ray_spp) / self.options.ray_split_size))
        qrx = torch.tensor_split(qrx, sections)
        qry = torch.tensor_split(qry, sections)
        radiance = zeros_like_vec(pix_grid, radiance_channels)
        alpha = zeros_like_vec(pix_grid, 1)
        extras_c = dict()
        extras_v = dict()
        for qrx, qry in zip(qrx, qry):
            espp = len(qrx)
            qrx, qry = qrx[..., None, None], qry[..., None, None]
            grid = pix_grid
            grid = float4(grid.x + (qrx - 0.5) * (2 / W), grid.y + (qry - 0.5) * (2 / H), grid.zw)  # espp, HW, 4
            grid = grid.reshape(-1, 4)
            rays_o, rays_d = self._view_dir_impl(grid, cam_v, cam_p, self.camera_VP())
            transfer = ones_like_vec(rays_o, radiance_channels)
            for d in range(self.options.ray_depth):
                t, i = raycaster.query(rays_o, rays_d, far)
                ray_outputs = sampler(rays_o, rays_d, t, i, d)
                radiance = radiance + (transfer * ray_outputs.radiance).view(espp, -1, radiance_channels).sum(0)
                alpha = alpha + ray_outputs.alpha.reshape(espp, -1, 1).sum(0)
                transfer = transfer * ray_outputs.transfer
                rays_o, rays_d = ray_outputs.next_rays_o, ray_outputs.next_rays_d
                if d == 0:
                    for k, v in ray_outputs.extras.items():
                        if k not in extras_c:
                            extras_c[k] = espp
                            extras_v[k] = v.reshape(espp, H, W, v.shape[-1]).sum(0)
                        else:
                            extras_c[k] += espp
                            extras_v[k] = extras_v[k] + v.reshape(espp, H, W, v.shape[-1]).sum(0)
        radiance = radiance.reshape(H, W, radiance_channels) / self.options.ray_spp
        alpha = saturate(alpha.reshape(H, W, 1) / self.options.ray_spp)
        for k in extras_v:
            extras_v[k] = torch.flipud(extras_v[k] / extras_c[k])
        return torch.flipud(radiance), torch.flipud(alpha), extras_v

    def pbr(self):
        return self.trace_rays(self.sampler_brdf)
