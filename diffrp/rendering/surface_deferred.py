import torch
import warnings
import threading
import nvdiffrast.torch as dr
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Union, List, Tuple, Callable, Optional

from .camera import Camera
from ..utils.cache import cached
from ..utils.shader_ops import *
from .mixin import RenderSessionMixin
from ..utils.colors import linear_to_srgb
from ..utils.geometry import point_in_tri2d
from ..scene import Scene, ImageEnvironmentLight
from ..utils.composite import alpha_blend, alpha_additive
from ..utils.coordinates import unit_direction_to_latlong_uv
from ..materials.base_material import SurfaceInput, SurfaceUniform, SurfaceOutputStandard
from .interpolator import FullScreenInterpolator, MaskedSparseInterpolator, polyfill_interpolate, float_to_triidx
from ..utils.light_transport import prefilter_env_map, pre_integral_env_brdf, irradiance_integral_env_map, fresnel_schlick_smoothness


tls = threading.local()


@dataclass
class SurfaceDeferredRenderSessionOptions:
    """
    Options for ``SurfaceDeferredRenderSession``.

    Args:
        max_layers (int):
            Maximum layers to be composed for semi-transparency.
            0 or negative values mean unlimited.
            Ignored if the render session is created in ``opaque_only`` mode (default).
            Otherwise, defaults to 0 (unlimited).
            There is a technical upper-bound of 32767 layers for composition.
            However, in almost all cases the layers are bounded by your GPU memory,
            and there will not be 32767 layers in any realistic scenes,
            so this value will never be reached.
        interpolator_impl (str):
            | One of 'stencil_masked' and 'full_screen'.
            | Specifies the implementation of the interpolator.
              Defaults to 'stencil_masked'.
            | For 'stencil_masked' interpolator, for each material,
              only the active pixels go through the material call.
              This is memory and compute efficient but introduces more
              GPU pipeline flushes between material calls.
              You will see a flattened batched pixels in your material ``shade`` function,
              which means the pixel input and output tensors will have shape (B, C),
              where B is number of flattened pixels and C is number of channels.
            | 'full_screen' is the legacy implementation from DiffRP ``0.0.1``.
              For each material, all pixels including pixels outside the material region
              that are discarded afterwards.
              Pixel input and output tensors in materials shall have shape (B, H, W, C).
              This implementation uses much more memory and compute,
              but may be slightly more efficient due to fewer GPU pipeline flushes
              when there is only one material.
        deterministic (bool):
            Try to make the render deterministic.
            Defaults to True.
            Usually the variance is low enough for all operations in this deferred pipeline,
            but in extraordinary cases if you want to accumulate results
            between different calls to reduce variance
            for sampling processes, you may turn it off.
        ibl_specular_samples (int):
            Sample count for specular lighting in PBR Image-based lighting (IBL).
            Defaults to 512.
            Higher samples can reduce bias for environment specular lighting,
            but takes linearly more memory.
        ibl_specular_base_resolution (int):
            Resolution for specular lighting in IBL,
            whose effect is mostly visible for mirrors (metallic 1, roughness 0).
            Defaults to 256.
            Memory is quadratic to this parameter.
        ibl_specular_mip_levels (int):
            Levels of mip-maps for IBL specular lighting.
            Defaults to 5.
            Can reduce bias for roughness levels if increased,
            but takes linearly more memory.
        ibl_diffuse_sample_resolution (int):
            Sample resolution of diffuse irradiance convolution in IBL.
            Defaults to 64.
            Higher resolution can make the diffuse lighting more accurate,
            but this is usually not necessary.
            Memory is quadratic to this parameter.
        ibl_diffuse_base_resolution (int):
            Resolution of diffuse irradiance convolution computation in IBL.
            Defaults to 16.
            Higher resolution can make the diffuse lighting more accurate,
            but this is usually not necessary.
            Memory is quadratic to this parameter.
    """
    max_layers: int = 0
    interpolator_impl: Literal['full_screen', 'stencil_masked'] = 'stencil_masked'

    deterministic: bool = True

    ibl_specular_samples: int = 512
    ibl_specular_base_resolution: int = 256
    ibl_specular_mip_levels: int = 5
    ibl_diffuse_sample_resolution: int = 64
    ibl_diffuse_base_resolution: int = 16

    def __post_init__(self):
        if self.max_layers <= 0:
            self.max_layers = 32767

    @property
    def intepolator_impl(self):
        warnings.warn("`intepolator_impl` is deprecated, please use `interpolator_impl` instead.")
        return self.interpolator_impl
    
    @intepolator_impl.setter
    def intepolator_impl(self, value):
        warnings.warn("`intepolator_impl` is deprecated, please use `interpolator_impl` instead.")
        self.interpolator_impl = value


class _EdgeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, frag_xy, flat_tris_xy, rast, rgb):
        ctx.save_for_backward(frag_xy, flat_tris_xy, rast, rgb)
        return rgb

    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output):
        frag_xy, flat_tris_xy, rast, rgb = ctx.saved_tensors
        # frag: H, W, 2
        # rgb: H, W, 3
        # rast: H, W, 4
        # flat_tris_xy: F, 3, 2
        h, w = frag_xy.shape[:2]
        tri_idx: torch.IntTensor = rast.w.int()
        is_unhit = tri_idx == 0
        scr_tris = flat_tris_xy[tri_idx.squeeze(-1) - 1]

        def _grad_y():
            # vertical
            grad_edge: torch.Tensor = -0.5 * dot(grad_output[:-1] + grad_output[1:], rgb[:-1] - rgb[1:])
            has_edge = tri_idx[:-1] != tri_idx[1:]  # H - 1, W, 1
            a_in_b = is_unhit[1:] | point_in_tri2d(frag_xy[:-1], *torch.unbind(scr_tris[1:], dim=-2))  # H - 1, W, 1
            b_in_a = is_unhit[:-1] | point_in_tri2d(frag_xy[1:], *torch.unbind(scr_tris[:-1], dim=-2))
            grad_a = (has_edge & a_in_b).float() * grad_edge
            grad_b = (has_edge & b_in_a).float() * grad_edge
            return (
                rst.supercat([grad_a, grad_a.new_zeros(1)], dim=0)
                + rst.supercat([grad_b.new_zeros(1), grad_b], dim=0)
            )
        
        grad_y = _grad_y()

        def _grad_x():
            # horizontal
            grad_edge: torch.Tensor = 0.5 * dot(grad_output[:, :-1] + grad_output[:, 1:], rgb[:, :-1] - rgb[:, 1:])
            has_edge = tri_idx[:, :-1] != tri_idx[:, 1:]
            a_in_b = is_unhit[:, 1:] | point_in_tri2d(frag_xy[:, :-1], *torch.unbind(scr_tris[:, 1:], dim=-2))
            b_in_a = is_unhit[:, :-1] | point_in_tri2d(frag_xy[:, 1:], *torch.unbind(scr_tris[:, :-1], dim=-2))
            grad_a = (has_edge & a_in_b).float() * grad_edge
            grad_b = (has_edge & b_in_a).float() * grad_edge
            return (
                rst.supercat([grad_a, grad_a.new_zeros(1)], dim=1)
                + rst.supercat([grad_b.new_zeros(1), grad_b], dim=1)
            )

        grad_x = _grad_x()

        grad_rgb = grad_output.clone()
        return torch.cat([grad_x * (w / 2), grad_y * (h / 2)], dim=-1), None, None, grad_rgb


class SurfaceDeferredRenderSession(RenderSessionMixin):
    """
    The heart of the deferred rasterization render pipeline in DiffRP.
    Most intermediate options in this class are cached,
    so you are recommended to create a new session
    whenever you change the scene and/or the camera,
    including rendering multiple frames of the same object,
    or changing attributes on the object.
    Initialization of this class is made light-weight and fast.

    It is thread-safe to run more than one sessions on the same GPU by default.

    DiffRP operations do not support multiple GPUs on a single process.
    You will need to access the GPUs with a distributed paradigm,
    where each process only uses one GPU.

    Args:
        scene (Scene): The scene to be rendered.
        camera (Camera): The camera to be rendered.
        opaque_only (bool): Render in opaque-only mode.
            Drops support for transparency, and forces all alpha values to be 1.
            This can save memory and time if you do not need transparency support.
            Defaults to True.
            You need to specify False if you need transparency.
        options (SurfaceDeferredRenderSessionOptions): Options for the session.
            See :py:class:`SurfaceDeferredRenderSessionOptions` for details.
        ctx (RasterizeContext): The backend context to be used.
            By default, each GPU and each thread has its own, shared context.
            This render pipeline uses ``nvdiffrast`` as the backend.
            You can pass your own ``RasterizeCudaContext`` or ``RasterizeGLContext`` if needed.
    """
    def __init__(
        self,
        scene: Scene,
        camera: Camera,
        opaque_only: bool = True,
        options: Optional[SurfaceDeferredRenderSessionOptions] = None,
        ctx: Optional[Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]] = None,
    ) -> None:
        self.scene = scene
        self.camera = camera
        self.opaque_only = opaque_only
        if options is None:
            options = SurfaceDeferredRenderSessionOptions()
        if opaque_only:
            options.max_layers = 1
        if ctx is not None:
            self.ctx = ctx
        else:
            device = torch.cuda.current_device()
            if not hasattr(tls, 'ctx_cache'):
                tls.ctx_cache = {}
            if device not in tls.ctx_cache:
                tls.ctx_cache[device] = dr.RasterizeCudaContext()
            self.ctx = tls.ctx_cache[device]
        self.options = options
    
    @cached
    def clip_space(self):
        """
        All vertices concatenated, transformed into the GL clip space for projection.

        Returns:
            torch.Tensor: Tensor of shape (V, 4).
        """
        vao = self.vertex_array_object()
        return transform_point(vao.world_pos, self.camera_VP()).contiguous()
    
    @cached
    def rasterize(self):
        """
        Rasterize the scene. Each layer is a tensor of shape (1, H, W, 4).
        xyzw means (u, v, z-depth, triangle_index).

        u, v is defined as the barycentric coordinates:
        
        :math:`u \\cdot v_1 + v \\cdot v_2 + (1 - u - v) \\cdot v_3 = p`.

        The layers in the front in the list are the layers in the front (closer to the camera) in 3D.

        Returns:
            List[torch.Tensor]: rasterized layers. Tensors of shape (1, H, W, 4).
        """
        vao = self.vertex_array_object()
        clip_space = self.clip_space()
        h, w = self.camera.resolution()
        r_layers: List[torch.Tensor] = []
        rng = torch.tensor([[0, len(vao.tris)]], dtype=torch.int32)
        with dr.DepthPeeler(self.ctx, clip_space, vao.tris.contiguous(), (h, w), rng) as dp:
            for i in range(self.options.max_layers):
                rast, rast_db = dp.rasterize_next_layer()
                if (i < self.options.max_layers - 1) and (rast.a <= 0).all():
                    break
                r_layers.append(rast)
        return r_layers

    @cached
    def layer_material(self) -> List[List[Tuple[SurfaceInput, SurfaceOutputStandard]]]:
        """
        Evaluate materials for the scene. Each layer is a list of tensors of material inputs/outputs.
        Shapes of each attribute depend on the interpolator implementation.
        See the documentation for ``SurfaceInput``, ``SurfaceOutputStandard`` and ``interpolator_impl``.
        """
        m_layers: List[List[Tuple[SurfaceInput, SurfaceOutputStandard]]] = []
        vao = self.vertex_array_object()
        r_layers = self.rasterize()
        cam_v = self.camera_V()
        cam_p = self.camera_P()
        for rast in r_layers:
            mats = []
            if self.options.interpolator_impl == 'stencil_masked':
                vi_idx = float_to_triidx(rast[..., -1])
                stencil_buf = vao.stencils[vi_idx]
            for pi, x in enumerate(self.scene.objects):
                su = SurfaceUniform(x.M, cam_v, cam_p)
                if self.options.interpolator_impl == 'full_screen':
                    si = SurfaceInput(su, vao, FullScreenInterpolator(rast, vao.tris))
                elif self.options.interpolator_impl == 'stencil_masked':
                    si = SurfaceInput(su, vao, MaskedSparseInterpolator(rast, vao.tris, stencil_buf == pi + 1))
                    if len(si.interpolator.indices[0]) == 0:
                        mats.append((si, SurfaceOutputStandard()))
                        continue
                so = x.material.shade(su, si)
                mats.append((si, so))
            m_layers.append(mats)
        return m_layers

    def gbuffer_collect(
        self,
        operator: Callable[[SurfaceInput, SurfaceOutputStandard], torch.Tensor],
        default: Union[List[float], torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Collect the screen G-buffers from material evaluation results.
        
        The results from the operator given are collected against all materials into
        screen-sized buffers, one per rasterized layer.

        Args:
            operator (callable):
                A function that takes two arguments, ``SurfaceInput`` and
                ``SurfaceOutputStandard`` as input, and outputs a tensor.
            default (List[float]):
                If the operator returns ``None``, the default value is filled for the material.
                The default value is also used for inactive pixels (pixels without geometry).
        
        Returns:
            List[torch.Tensor]: Layered G-buffers collected.
        """
        vao = self.vertex_array_object()
        h, w = self.camera.resolution()
        gbuffers = []
        if self.options.interpolator_impl == 'stencil_masked':
            if default == [default[0]] * len(default):
                defaults = torch.full([1, h, w, len(default)], default[0], dtype=torch.float32, device='cuda')
            else:
                defaults = gpu_f32(default).expand(1, h, w, len(default))
            return [
                self._gbuffer_collect_layer_impl_stencil_masked(mats, operator, defaults)
                for mats in self.layer_material()
            ]
        assert self.options.interpolator_impl == 'full_screen', "Unrecognized `interpolator_impl` '%s'" % self.options.interpolator_impl
        for rast, mats in zip(self.rasterize(), self.layer_material()):
            buffer = [gpu_f32(default).expand(1, h, w, len(default))]
            stencil_lookup = [0]
            for si, so in mats:
                op = operator(si, so)
                if op is None:
                    stencil_lookup.append(0)
                else:
                    stencil_lookup.append(len(buffer))
                    buffer.append(op)
            stencil_lookup = vao.stencils.new_tensor(stencil_lookup)
            gbuffers.append(torch.gather(torch.cat(buffer), 0, stencil_lookup[vao.stencils][float_to_triidx(rast.a)].repeat_interleave(len(default), dim=-1).long()))
        return gbuffers

    def compose_layers(
        self,
        colors: list, alphas: Optional[list] = None,
        blend_fn: Optional[Callable] = None,
        return_alpha: bool = True
    ) -> torch.Tensor:
        """
        Compose multiple layers into one.
        The layers in the front in the lists are the layers in the front in 3D.

        Args:
            colors (List[torch.Tensor]):
                RGB colors, or arbitrary values to be blended.
                Tensors can have arbitrary shapes, but consistent across the list.
            alphas (List[torch.Tensor]):
                Alphas. Tensors should have matching shapes with ``colors``,
                with the last dimension (channels) equal to 1.
                Defaults to alphas of layers in the current scene.
            blend_fn (callable):
                A function that accepts 4 arguments: (fg_value, fg_alpha, bg_value, bg_alpha)
                and returns a tuple of composition results (value, alpha).
                Defaults to alpha blending.
            return_alpha (bool):
                Whether to include the alpha channel in the result.
                The alpha channel will be concatenated to the end if set.
        """
        if alphas is None:
            alphas = self.alpha_layered()
        if blend_fn is None:
            blend_fn = alpha_blend
        if len(colors) == 0:
            return torch.zeros(*self.camera.resolution(), 4, dtype=torch.float32, device='cuda')
        frame_buffer = colors[-1]
        frame_alpha = alphas[-1]
        for g, a in zip(reversed(colors[:-1]), reversed(alphas[:-1])):
            frame_buffer, frame_alpha = blend_fn(frame_buffer, frame_alpha, g, a)
        return torch.flipud((floatx(frame_buffer, frame_alpha) if return_alpha else frame_buffer).squeeze(0))

    @cached
    def alpha_layered(self) -> List[torch.Tensor]:
        """
        Returns:
            List[torch.Tensor]: Alphas of layers in the current scene.
        """
        if self.opaque_only:
            return self.gbuffer_collect(lambda x, y: ones_like_vec(x.interpolator.vi_data, 1), [0.])
        else:
            return self.gbuffer_collect(lambda x, y: y.alpha if y.alpha is not None else ones_like_vec(x.interpolator.vi_data, 1), [0.])

    @cached
    def albedo_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: y.albedo, [1.0, 0.0, 1.0])

    @cached
    def emission_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: y.emission, [0.0, 0.0, 0.0])

    def _collector_mso(self, si: SurfaceInput, so: SurfaceOutputStandard):
        if so.metallic is None and so.smoothness is None and so.occlusion is None:
            return None
        return float3(
            so.metallic if so.metallic is not None else 0.0,
            so.smoothness if so.smoothness is not None else 0.5,
            so.occlusion if so.occlusion is not None else 1.0
        )

    @cached
    def mso_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(self._collector_mso, [0.0, 0.5, 1.0])

    @cached
    def world_space_normal_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(self._collector_world_normal, [0.0, 0.0, 0.0])

    @cached
    def camera_space_normal_layered(self) -> List[torch.Tensor]:
        world_normals = self.world_space_normal_layered()
        return [
            normalized(transform_vector3x3(world_normal, self.camera.V()))
            for world_normal in world_normals
        ]

    @cached
    def depth_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: -transform_vector3x3(x.world_pos, self.camera.V()).z, [0.])

    @cached
    def distance_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: length(x.world_pos - x.uniforms.camera_position), [0.])

    @cached
    def world_position_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: x.world_pos, [0., 0., 0.])

    @cached
    def local_position_layered(self) -> List[torch.Tensor]:
        return self.gbuffer_collect(lambda x, y: x.local_pos, [0., 0., 0.])

    @cached
    def albedo(self):
        """
        Returns:
            torch.Tensor:
                Rendered albedo, or base color RGBA, in linear space.
                Albedo from transparent objects are alpha-blended if ``opaque_only`` is disabled.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.albedo_layered())

    @cached
    def albedo_srgb(self):
        """
        Returns:
            torch.Tensor:
                Rendered albedo, or base color RGBA. RGBs are in sRGB space.
                Albedo from transparent objects are alpha-blended in linera space if ``opaque_only`` is disabled.
                Tensor of shape (H, W, 4).
        """
        albedo = self.albedo()
        return float4(linear_to_srgb(albedo.rgb), albedo.a)

    @cached
    def emission(self):
        """
        Returns:
            torch.Tensor:
                Rendered direct emission RGB, in linear space.
                Emission is composed additively for transparent objects if ``opaque_only`` is disabled.
                Tensor of shape (H, W, 3).
        """
        return self.compose_layers(self.emission_layered(), blend_fn=alpha_additive, return_alpha=False)

    @cached
    def false_color_mask_mso(self):
        """
        Returns:
            torch.Tensor:
                Rendered Metallic, Smoothness and ambiend Occlusion values
                linearly mapped to RGB channels.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.mso_layered())

    @cached
    def world_space_normal(self):
        """
        Returns:
            torch.Tensor:
                World-space normals. Raw values of unit vectors.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.world_space_normal_layered())

    @cached
    def camera_space_normal(self):
        """
        Returns:
            torch.Tensor:
                Camera-space normals ([0, 0, 1] is directly facing the camera).
                Raw values of unit vectors.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.camera_space_normal_layered())

    @cached
    def false_color_world_space_normal(self):
        """
        Returns:
            torch.Tensor:
                World-space normals XYZ linearly mapped to RGB space from [-1, 1] to [0, 1].
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        n = self.world_space_normal()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def false_color_camera_space_normal(self):
        """
        Returns:
            torch.Tensor:
                Camera-space normals XYZ linearly mapped to RGB space from [-1, 1] to [0, 1],
                the common blue-ish rendering of scene normals.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        n = self.camera_space_normal()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def local_position(self):
        """
        Returns:
            torch.Tensor:
                Object-space positions XYZ raw values.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.local_position_layered())

    @cached
    def world_position(self):
        """
        Returns:
            torch.Tensor:
                World-space positions XYZ raw values.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        return self.compose_layers(self.world_position_layered())

    @cached
    def false_color_nocs(self):
        """
        Note: Objects are NOT automatically normalized before this rendering!

        Returns:
            torch.Tensor:
                Object-space positions XYZ linearly mapped to RGB space from [-1, 1] to [0, 1],
                conformal with the NOCS definition.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, 4).
        """
        n = self.local_position()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def depth(self):
        """
        Returns:
            torch.Tensor:
                Depth of the scene, raw values.
                Depth is distance to the camera-z plane.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Emptiness results in zero depth.
                Tensor of shape (H, W, 1).
        """
        return self.compose_layers(self.depth_layered(), return_alpha=False)

    @cached
    def distance(self):
        """
        Returns:
            torch.Tensor:
                Distance of the scene to the camera, raw values.
                Has a similar effect but slightly different definition to ``depth``.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Emptiness results in zero distance.
                Tensor of shape (H, W, 1).
        """
        return self.compose_layers(self.distance_layered(), return_alpha=False)
    
    def aov(self, key: str, bg_value: List[float]):
        """
        Args:
            key (str): The key you want to collect from material AOV outputs.
            bg_value (List[float]):
                Values to fill in where the AOV is either
                not specified in materials or background is hit.

        Returns:
            torch.Tensor:
                Collected full-screen buffers of AOVs.
                Values are alpha-blended for transparent objects, if ``opaque_only`` is disabled.
                Alpha channel is geometry transparency.
                Tensor of shape (H, W, C + 1).
        
        See also:
            :py:class:`diffrp.materials.base_material.SurfaceOutputStandard`
        """
        return self.compose_layers(self.gbuffer_collect(lambda x, y: y.aovs and y.aovs.get(key), bg_value))

    @cached
    def prepare_ibl(self):
        """
        Prepares the cache (prefiltered mips) for IBL.

        The return value can be provided via ``set_prepare_ibl`` for other Sessions
        if any ``ImageEnvironmentLight`` in the scene haven't been changed and are not requiring gradient
        to improve render performance.
        """
        opt = self.options
        prev_pre_levels = None
        prev_irradiance = None
        for light in self.scene.lights:
            if isinstance(light, ImageEnvironmentLight):
                image_rh = light.image_rh()
                pre_levels = prefilter_env_map(
                    image_rh,
                    deterministic=opt.deterministic,
                    base_resolution=opt.ibl_specular_base_resolution,
                    num_levels=opt.ibl_specular_mip_levels,
                    num_samples=opt.ibl_specular_samples
                )
                irradiance = irradiance_integral_env_map(
                    image_rh,
                    premip_resolution=opt.ibl_diffuse_sample_resolution,
                    base_resolution=opt.ibl_diffuse_base_resolution
                )
                if prev_pre_levels is not None:
                    pre_levels = prev_pre_levels + pre_levels
                if prev_irradiance is not None:
                    irradiance = prev_irradiance + irradiance
                prev_pre_levels = pre_levels
                prev_irradiance = irradiance
        return pre_levels, irradiance

    def set_prepare_ibl(self, cache):
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[SurfaceDeferredRenderSession.prepare_ibl.__qualname__] = cache

    @staticmethod
    @torch.jit.script
    def _ibl_combine_impl(
        metal: torch.Tensor, smooth: torch.Tensor, albedo: torch.Tensor,
        irradiance_fetch: torch.Tensor, specular_fetch: torch.Tensor,
        brdf: torch.Tensor, n_dot_v: torch.Tensor, ao: torch.Tensor
    ):
        dielectric = 1 - metal
        f0 = fma(albedo, metal, 0.04 * dielectric)
        f = fresnel_schlick_smoothness(n_dot_v, f0, smooth)
        bx, by = brdf[..., 0:1], brdf[..., 1:2]

        diffuse = irradiance_fetch * albedo * dielectric * (1 - f)
        specular_diffuse = fma(specular_fetch, fma(f, bx, by), diffuse)
        return specular_diffuse * ao

    def ibl(
        self,
        world_normal: torch.Tensor,
        view_dir: torch.Tensor,
        mso: torch.Tensor,
        albedo: torch.Tensor
    ):
        """
        Evaluate Image-based Lighting (IBL) given the necessary G-buffers.

        Uses split-sum approximation and pre-filtering techniques.
        """
        env_brdf = pre_integral_env_brdf()
        pre_levels, irradiance = self.prepare_ibl()

        n_dot_v = torch.relu_(-dot(world_normal, view_dir))
        r = reflect(view_dir, world_normal)
        n_uv = float2(*unit_direction_to_latlong_uv(world_normal))

        metal, smooth, ao = mso.x, mso.y, mso.z
        r_uvm = float3(*unit_direction_to_latlong_uv(r), smooth)
        pre_fetch = sample3d(pre_levels, r_uvm, 'latlong')

        brdf = sample2d(env_brdf, float2(n_dot_v, smooth))
        irradiance_fetch = sample2d(irradiance, n_uv, 'latlong')

        return SurfaceDeferredRenderSession._ibl_combine_impl(
            metal, smooth, albedo, irradiance_fetch, pre_fetch, brdf, n_dot_v, ao
        )

    @cached
    def pbr_layered(self) -> List[torch.Tensor]:
        return [
            self.ibl(world_normal, self.view_dir(), mso, albedo)
            for world_normal, mso, albedo in zip(
                self.world_space_normal_layered(),
                self.mso_layered(),
                self.albedo_layered()
            )
        ]

    @cached
    def pbr(self):
        """
        Physically-based rendering.

        Returns:
            torch.Tensor:
                Rendered HDR RGBA image in linear space.
                Colors from transparent objects are alpha-blended if ``opaque_only`` is disabled.
                Tensor of shape (H, W, 4).
        """
        if len(self.scene.lights) == 0:
            raise ValueError("PBR cannot be computed when there is no light in the scene!")
        skybox = 0
        any_skybox = False
        for light in self.scene.lights:
            if isinstance(light, ImageEnvironmentLight) and light.render_skybox:
                any_skybox = True
                skybox = skybox + sample2d(
                    light.image_rh(),
                    float2(*unit_direction_to_latlong_uv(self.view_dir()))
                )
        if any_skybox:
            return self.compose_layers(
                self.pbr_layered() + [skybox],
                self.alpha_layered() + [ones_like_vec(skybox, 1)]
            )
        else:
            return self.compose_layers(self.pbr_layered())

    def nvdr_antialias(self, rgb: torch.Tensor):
        """
        Applies the anti-aliasing technique in ``nvdiffrast`` to an image.
        This can generate visibility gradients in differentiable rendering.
        
        Semi-transparent objects are not well-supported by this operation.
        
        Args:
            rgb (torch.Tensor):
                Tensor of shape (H, W, 3). RGB image to be anti-aliased.
                Must be rendered in this session.
        Returns:
            torch.Tensor: Anti-aliased image.
        """
        rgb = torch.flipud(rgb)[None].contiguous()
        rast = self.rasterize()[0].contiguous()
        vao = self.vertex_array_object()
        return torch.flipud(dr.antialias(rgb, rast, self.clip_space(), vao.tris.contiguous()).squeeze(0))

    def edge_gradient(self, rgb: torch.Tensor):
        """
        Applies the edge-gradient technique in
        *Rasterized Edge Gradients: Handling Discontinuities Differentiably*.
        This can generate visibility gradients in differentiable rendering.

        The image will be returned as-is in the forward session,
        but gradients can start flowing through geometry visibility.

        Semi-transparent objects are not well-supported by this operation.
        
        Args:
            rgb (torch.Tensor):
                Tensor of shape (H, W, 3). RGB image to compute edge gradient with.
                Must be rendered in this session.
        Returns:
            torch.Tensor: the same RGB image, but with gradient operators attached.
        """
        rast = self.rasterize()[0].squeeze(0)
        clip_space = self.clip_space()
        tris = self.vertex_array_object().tris

        frag = polyfill_interpolate(clip_space, rast, tris[float_to_triidx(rast.w.squeeze(-1)) - 1], 1.0)
        frag_xy = frag.xy / frag.w

        flat_tris_xy = (clip_space.xy / clip_space.w)[tris]  # F, 3, 2

        return torch.flipud(_EdgeGradient.apply(frag_xy, flat_tris_xy, rast, torch.flipud(rgb)))
