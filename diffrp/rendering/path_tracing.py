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
    """
    Options for ``PathTracingSession``.

    Args:
        ray_depth (int):
            The tracing depth of a ray.
            Defaults to 3, meaning at maximum 3 bounces of a ray will be computed.
            At the end of the bounce, ``pbr_ray_last_bounce`` determines the behavior of the ray if you are using the default ``sampler_brdf``.
            Otherwise, the behavior is defined by your implementation of the sampler.
        ray_spp (int):
            Samples per pixel. Defaults to 16.
        ray_split_size (int):
            Split the rays traced as a batch.
            Useful to reduce memory consumption when gradients are not required.
            Defaults to 8M (8 x 1024 x 1024) rays.
        deterministic (bool):
            Try to make the render deterministic.
            Defaults to True.
        pbr_ray_step_epsilon (float):
            To avoid near-intersection at the origin (surface bounce point) of bounce rays, we march ray origins by this amount when generating them in ``sampler_brdf``.
            You are also recommended to reuse this value for your own sampler for bounce rays.
            Defaults to 1e-3 (0.001).
        pbr_ray_last_bounce (str):
            One of 'void' and 'skybox'.
            Special to ``sampler_brdf`` in PBR rendering.
            If set to 'void' (default), a bounce ray ends with zero radiance if it has not reached the sky in ``ray_depth`` bounces.
            If set to 'skybox', it ends with the sky radiance in the final-bounce direction instead.
        raycaster_impl (str):
            | One of 'brute-force', 'naive-pbbvh' and 'torchoptix'.
              Defaults to 'torchoptix'.
            | The implementation for scene raycaster.
            | 'torchoptix' requires ``torchoptix`` to be installed. It uses hardware RT cores for raycasting. This is the fastest implementation.
            | 'naive-pbbvh' uses a software implementation of a GPU BVH. It is usually 10x to 100x slower than the hardware solution of 'torchoptix'.
            | 'brute-force' uses a software brute-force ray-triangle tests comparing all pairs of rays and triangles. It is fast if there is little geometry (fewer than ~10 triangles).
        raycaster_epsilon (float):
            Only relevant for 'brute-force' and 'naive-pbbvh' raycaster implementations. Discards triangles parallel to the rays beyond this threshold.
            Defaults to 1e-8.
        raycaster_builder (str):
            | One of 'morton' and 'splitaxis'.
              Defaults to 'splitaxis'.
            | The BVH builder. Only relevant for 'naive-pbbvh' raycaster implementation.
            | 'morton' uses a linear BVH sorting triangles based on 30-bit morton code. This has lower query performance but faster building.
            | 'splitaxis' sorts and splits triangles based on the longest axis at each BVH level.
        optix_log_level (int):
            OptiX log level, 0 to 4. Higher means more verbose.
            Defaults to 3.
    """
    ray_depth: int = 3
    ray_spp: int = 16
    ray_split_size: int = 8 * 1024 * 1024

    deterministic: bool = True
    
    pbr_ray_step_epsilon: float = 1e-3
    pbr_ray_last_bounce: Literal['void', 'skybox'] = 'void'

    raycaster_impl: Literal['brute-force', 'naive-pbbvh', 'torchoptix'] = 'torchoptix'
    raycaster_epsilon: float = 1e-8
    raycaster_builder: Literal['morton', 'splitaxis'] = 'splitaxis'
    optix_log_level: Literal[0, 1, 2, 3, 4] = 3


@dataclass
class RayOutputs:
    """
    The protocol outputs for path tracing sampler.
    
    Args:
        radiance (torch.Tensor):
            Shape (R, C) where R is number of rays for the current trace and C is specified channels in ``trace_rays``.
            Radiance value of the current hit.
            The output radiance would be the sum of radiance times the attenuation at the radiance.
        transfer (torch.Tensor):
            Should be brocastable with radiance.
            Transfer value of the current hit.
            The total attenuation value would be the product of transfer values.
            The attenuation value for the current radiance excludes the current transfer function (starts with all-1s for the first-hit radiance).
        next_rays_o (torch.Tensor):
            Shape (R, 3), bounce ray origins.
            You may want to advance ray origins by a small value (e.g., ``pbr_ray_step_epsilon``) to avoid intersecting the same point at the origin for secondary bounces.
        next_rays_d (torch.Tensor):
            Shape (R, 3), bounce ray directions.
        alpha (torch.Tensor):
            Shape (R, 1), alpha values.
            Additively composed and saturated for a hit-mask output.
            Has no exact physical meaning for different values between 0 and 1 in path tracing,
            and has no influence on radiance composition.
        extras (Dict[str, torch.Tensor]):
            Extra auxiliary outputs indexed by a string name.
            Collected on the first hit and ignoring .
            Useful for auxiliary outputs like normals, base colors and AOVs.
    See also:
        :py:meth:`diffrp.rendering.path_tracing.PathTracingSession.trace_rays`
    """
    radiance: torch.Tensor
    transfer: torch.Tensor
    next_rays_o: torch.Tensor
    next_rays_d: torch.Tensor
    alpha: torch.Tensor
    extras: Dict[str, torch.Tensor]


class PathTracingSession(RenderSessionMixin):
    """
    Path tracing render pipeline in DiffRP.
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
            cfg['optix_log_level'] = self.options.optix_log_level
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

    def sampler_brdf(self, rays_o, rays_d, t, i, d: int) -> RayOutputs:
        """
        A *sampler* for ``trace_rays`` implementing the principled PBR BRDF.
        
        See :py:class:`RayOutputs` and :py:meth:`~diffrp.rendering.path_tracing.PathTracingSession.pbr` for semantics of outputs.
        
        See also:
            :py:meth:`~diffrp.rendering.path_tracing.PathTracingSession.trace_rays` for meaning of arguments.
        """
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
        """
        Trace the ray paths.
        
        Note that missed rays are not automatically excluded in further traces.
        You may need to specify a transfer value of zeros to avoid light leakage.
        
        Args:
            sampler (Callable):
                | Sampler function to produce radiance, alpha and extra values.
                  Take five arguments ``(rays_o, rays_d, t, i, d)`` and outputs :py:class:`RayOutputs`.
                | ``rays_o`` and ``rays_d`` are of shape (R, 3), and defines current tracing rays.
                | ``t`` of shape (R) is the distance from the ray origin to the hit point. It is guaranteed to be ``camera_far()`` if not hit.
                | ``i`` of shape (R) is the primitive index of the hit point, starting from 0. The value is undefined for non-hit rays.
                | Usually you do not need to handle the semantics of ``i`` yourselves, but
                  use ``layer_material_rays`` and ``_gbuffer_collect_layer_impl_stencil_masked`` to collect information about the scene.
                | ``d`` is the current ray depth. ``0`` is the first rays emitting from the camera.
                | Returns :py:class:`RayOutputs`. See the documentation for :py:class:`RayOutputs` for meanings of components.
            radiance_channels (int):
                Number of channels for radiance outputs in sampler.
                Defaults to 3.

        Returns:
            Tuple of (radiance, alpha, extras):
                ``radiance`` is a tensor of shape (H, W, ``radiance_channels``).
                ``alpha`` is a tensor of shape (H, W, 1).
                ``extras`` is a dict from names to tensors of auxiliary attributes,
                all of which has shape (H, W, C) where C is the number of channels output by the ``sampler``.
        """
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
        """
        Path-traced PBR rendering.
        
        Returns:
            Tuple of (radiance, alpha, extras):
                | ``radiance`` is a linear HDR RGB tensor of shape (H, W, 3).
                | ``alpha`` is a mask tensor of shape (H, W, 1).
                | ``extras`` is a dict from names to tensors of auxiliary attributes.
                  The included attributes are: ``albedo``, ``emission``, ``world_normal`` and ``world_position``.
                  ``world_normal`` are raw vectors in range of [-1, 1] and are not mapped to false colors.
                  All of them have shape (H, W, 3).
        """
        return self.trace_rays(self.sampler_brdf)
