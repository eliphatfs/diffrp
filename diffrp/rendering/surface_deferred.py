import torch
import nvdiffrast.torch as dr
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Union, List, Tuple, Callable, Optional
from .camera import Camera
from ..utils.cache import cached
from ..utils.shader_ops import *
from ..utils.composite import alpha_blend
from ..scene import Scene, ImageEnvironmentLight
from ..utils.coordinates import unit_direction_to_latlong_uv, near_plane_ndc_grid
from .interpolator import FullScreenInterpolator, MaskedSparseInterpolator
from ..materials.base_material import VertexArrayObject, SurfaceInput, SurfaceUniform, SurfaceOutputStandard
from ..utils.light_transport import prefilter_env_map, pre_integral_env_brdf, irradiance_integral_env_map, fresnel_schlick_roughness


@dataclass
class SurfaceDeferredRenderSessionOptions:
    max_layers: int = 0
    intepolator_impl: Literal['full_screen', 'stencil_masked'] = 'stencil_masked'
    ibl_specular_samples: int = 1024

    def __post_init__(self):
        if self.max_layers <= 0:
            self.max_layers = 16383


class SurfaceDeferredRenderSession:
    def __init__(
        self,
        ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext],
        scene: Scene,
        camera: Camera,
        opaque_only: bool = True,
        options: Optional[SurfaceDeferredRenderSessionOptions] = None
    ) -> None:
        self.ctx = ctx
        self.scene = scene
        self.camera = camera
        self.opaque_only = opaque_only
        if options is None:
            options = SurfaceDeferredRenderSessionOptions()
        if opaque_only:
            options.max_layers = 1
        self.options = options

    def _checked_cat(self, attrs: List[torch.Tensor], verts_ref: List[torch.Tensor], expected_dim):
        for v, a in zip(verts_ref, attrs):
            assert len(v) == len(a), "attribute length not the same as number of vertices"
            if isinstance(expected_dim, int):
                assert a.size(-1) == expected_dim, "expected %d dims but got %d for vertex attribute" % (expected_dim, a.size(-1))
        return torch.cat(attrs)

    @cached
    def vertex_array_object(self) -> VertexArrayObject:
        world_pos = []
        tris = []
        sts = [torch.full([1], 0, device='cuda', dtype=torch.int16)]
        objs = self.scene.objects
        offset = 0
        for s, x in enumerate(objs):
            world_pos.append(transform_point4x3(x.verts, x.M))
            assert x.tris.size(-1) == 3, "Expected 3 vertices per triangle, got %d" % x.tris.size(-1)
            tris.append(x.tris + offset)
            sts.append(x.tris.new_full([len(x.tris)], s + 1, dtype=torch.int16))
            offset += len(x.verts)
        total_custom_attrs = set().union(*(obj.custom_attrs.keys() for obj in objs))
        for k in total_custom_attrs:
            for obj in objs:
                if k in obj.custom_attrs:
                    sz = obj.custom_attrs[k].size(-1)
                    break
            else:
                assert False, "Internal assertion failure. You should never reach here."
            for obj in objs:
                if k in obj.custom_attrs:
                    assert len(obj.custom_attrs[k]) == len(obj.verts), "Attribute length not the same as number of vertices: %s" % k
                    assert obj.custom_attrs[k].size(-1) == sz, "Different dimensions for same custom attribute: %d != %d" % (obj.custom_attrs[k].size(-1), sz)
                else:
                    obj.custom_attrs[k] = zeros_like_vec(obj.verts, sz)

        verts_ref = [obj.verts for obj in objs]
        return VertexArrayObject(
            self._checked_cat([obj.verts for obj in objs], verts_ref, 3),
            self._checked_cat([obj.normals for obj in objs], verts_ref, 3),
            torch.cat(world_pos),
            torch.cat(tris).int().contiguous(),
            torch.cat(sts).int().contiguous(),
            self._checked_cat([obj.color for obj in objs], verts_ref, None),
            self._checked_cat([obj.uv for obj in objs], verts_ref, 2),
            self._checked_cat([obj.tangents for obj in objs], verts_ref, 4),
            {k: torch.cat([obj.custom_attrs[k] for obj in objs]) for k in total_custom_attrs}
        )
    
    @cached
    def clip_space(self):
        vao = self.vertex_array_object()
        camera = self.camera
        v = camera.V()
        p = camera.P()
        return transform_point(vao.world_pos, p @ v).contiguous()
    
    @cached
    def rasterize(self):
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
    def layer_material(self):
        m_layers: List[List[Tuple[SurfaceInput, SurfaceOutputStandard]]] = []
        vao = self.vertex_array_object()
        r_layers = self.rasterize()
        cam = self.camera
        for rast in r_layers:
            mats = []
            if self.options.intepolator_impl == 'stencil_masked':
                stencil_buf = vao.stencils[rast.a.int().squeeze(-1)]
            for pi, x in enumerate(self.scene.objects):
                su = SurfaceUniform(x.M, cam.V(), cam.P())
                if self.options.intepolator_impl == 'full_screen':
                    si = SurfaceInput(su, vao, FullScreenInterpolator(rast, vao.tris))
                elif self.options.intepolator_impl == 'stencil_masked':
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
    ):
        vao = self.vertex_array_object()
        h, w = self.camera.resolution()
        gbuffers = []
        for rast, mats in zip(self.rasterize(), self.layer_material()):
            if self.options.intepolator_impl == 'full_screen':
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
                gbuffers.append(torch.gather(torch.cat(buffer), 0, stencil_lookup[vao.stencils][rast.a.int()].repeat_interleave(len(default), dim=-1).long()))
            elif self.options.intepolator_impl == 'stencil_masked':
                buffer = gpu_f32(default).expand(1, h, w, len(default))
                indices = []
                values = []
                nind = 0
                for si, so in mats:
                    op = operator(si, so)
                    interpolator: MaskedSparseInterpolator = si.interpolator
                    if op is not None:
                        nind = len(interpolator.indices)
                        indices.append(interpolator.indices)
                        values.append(op)
                if nind != 0:
                    indices = tuple(torch.cat([x[i] for x in indices]) for i in range(nind))
                    values = torch.cat(values)
                    buffer = buffer.index_put(indices, values)
                gbuffers.append(buffer)
        return gbuffers

    def compose_layers(
        self,
        colors: list, alphas: Optional[list] = None,
        blend_fn: Optional[Callable] = None,
        return_alpha: bool = True
    ):
        if alphas is None:
            alphas = self.alpha_layered()
        if blend_fn is None:
            blend_fn = alpha_blend
        frame_buffer = colors[-1]
        frame_alpha = alphas[-1]
        for g, a in zip(reversed(colors[:-1]), reversed(alphas[:-1])):
            frame_buffer, frame_alpha = blend_fn(frame_buffer, frame_alpha, g, a)
        return torch.flipud((floatx(frame_buffer, frame_alpha) if return_alpha else frame_buffer).squeeze(0))

    @cached
    def alpha_layered(self):
        if self.opaque_only:
            return self.gbuffer_collect(lambda x, y: ones_like_vec(x.interpolator.vi_data, 1), [0.])
        else:
            return self.gbuffer_collect(lambda x, y: y.alpha if y.alpha is not None else ones_like_vec(x.interpolator.vi_data, 1), [0.])

    @cached
    def albedo_layered(self):
        return self.gbuffer_collect(lambda x, y: y.albedo, [1.0, 0.0, 1.0])

    @cached
    def emission_layered(self):
        return self.gbuffer_collect(lambda x, y: y.emission, [0.0, 0.0, 0.0])

    def collector_mso(self, si: SurfaceInput, so: SurfaceOutputStandard):
        if so.metallic is None and so.smoothness is None and so.occlusion is None:
            return None
        return float3(
            so.metallic if so.metallic is not None else 0.0,
            so.smoothness if so.smoothness is not None else 0.5,
            so.occlusion if so.occlusion is not None else 1.0
        )

    @cached
    def mso_layered(self):
        return self.gbuffer_collect(self.collector_mso, [0.0, 0.5, 1.0])

    def collector_world_normal(self, si: SurfaceInput, so: SurfaceOutputStandard):
        if so.normal is None:
            return si.world_normal
        if so.normal_space == 'tangent':
            vn = si.world_normal_unnormalized
            vnt = so.normal
            vt, vs = si.world_tangent.xyz, si.world_tangent.w
            vb = vs * cross(vn, vt)
            return normalized(vnt.x * vt + vnt.y * vb + vnt.z * vn)
        if so.normal_space == 'object':
            return normalized(transform_vector3x3(so.normal, si.uniforms.M))
        if so.normal_space == 'world':
            return normalized(so.normal)
        assert False, "Unknown normal space: " + so.normal_space

    @cached
    def world_space_normal_layered(self):
        return self.gbuffer_collect(self.collector_world_normal, [0.0, 0.0, 0.0])

    @cached
    def camera_space_normal_layered(self):
        world_normals = self.world_space_normal_layered()
        return [
            normalized(transform_vector3x3(world_normal, self.camera.V()))
            for world_normal in world_normals
        ]

    @cached
    def depth_layered(self):
        return self.gbuffer_collect(lambda x, y: -transform_vector3x3(x.world_pos, self.camera.V()).z, [0.])

    @cached
    def distance_layered(self):
        return self.gbuffer_collect(lambda x, y: length(x.world_pos - x.uniforms.camera_position), [0.])

    @cached
    def world_position_layered(self):
        return self.gbuffer_collect(lambda x, y: x.world_pos, [0., 0., 0.])

    @cached
    def local_position_layered(self):
        return self.gbuffer_collect(lambda x, y: x.local_pos, [0., 0., 0.])

    @cached
    def view_dir(self):
        grid = near_plane_ndc_grid(*self.camera.resolution(), torch.float32, 'cuda')
        grid = transform_point(grid, torch.linalg.inv(self.camera.P()))
        grid = grid.xyz / grid.w
        camera_matrix = torch.linalg.inv(self.camera.V())
        grid = transform_point(grid, camera_matrix)
        grid = grid.xyz / grid.w
        return normalized(grid - camera_matrix[:3, 3])

    @cached
    def albedo(self):
        return self.compose_layers(self.albedo_layered())

    @cached
    def emission(self):
        return self.compose_layers(self.emission_layered())

    @cached
    def false_color_mask_mso(self):
        return self.compose_layers(self.mso_layered())

    @cached
    def world_space_normal(self):
        return self.compose_layers(self.world_space_normal_layered())

    @cached
    def camera_space_normal(self):
        return self.compose_layers(self.camera_space_normal_layered())

    @cached
    def false_color_world_space_normal(self):
        n = self.world_space_normal()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def false_color_camera_space_normal(self):
        n = self.camera_space_normal()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def local_position(self):
        return self.compose_layers(self.local_position_layered())

    @cached
    def world_position(self):
        return self.compose_layers(self.world_position_layered())

    @cached
    def false_color_nocs(self):
        n = self.local_position()
        return float4(n.xyz * 0.5 + 0.5, n.a)

    @cached
    def depth(self):
        return self.compose_layers(self.depth_layered(), return_alpha=False)

    @cached
    def distance(self):
        return self.compose_layers(self.distance_layered(), return_alpha=False)

    def ibl(
        self,
        light: ImageEnvironmentLight,
        world_normal: torch.Tensor,
        view_dir: torch.Tensor,
        mso: torch.Tensor,
        albedo: torch.Tensor
    ):
        env_brdf = pre_integral_env_brdf()
        image_rh = light.image_rh()
        pre_levels = prefilter_env_map(image_rh, deterministic=True)
        irradiance = irradiance_integral_env_map(image_rh)

        n_dot_v = torch.relu(dot(world_normal, -view_dir))
        r = reflect(view_dir, world_normal)
        r_uv = float2(*unit_direction_to_latlong_uv(r))
        n_uv = float2(*unit_direction_to_latlong_uv(world_normal))

        pre_lod = (len(pre_levels) - 1) * saturate(1 - mso.y)
        pre_fetch = 0
        for i, level in enumerate(pre_levels):
            pre_level = torch.relu(sample2d(level, r_uv, 'latlong'))
            pre_weight = torch.relu(1 - torch.abs(pre_lod - i))
            pre_fetch = pre_fetch + pre_level * pre_weight

        brdf = sample2d(env_brdf, float2(n_dot_v, mso.y))
        irradiance_fetch = sample2d(irradiance, n_uv, 'latlong')

        f0 = 0.04 * (1.0 - mso.x) + albedo * mso.x
        f = fresnel_schlick_roughness(n_dot_v, f0, 1 - mso.y)

        specular = pre_fetch * (f * brdf.x + brdf.y)
        diffuse = irradiance_fetch * albedo * (1.0 - mso.x) * (1 - f)
        return (specular + diffuse) * mso.z
    
    def pbr_layer(
        self,
        world_normal: torch.Tensor,
        view_dir: torch.Tensor,
        mso: torch.Tensor,
        albedo: torch.Tensor
    ):
        render = torch.zeros_like(albedo)
        for light in self.scene.lights:
            if isinstance(light, ImageEnvironmentLight):
                render = render + self.ibl(light, world_normal, view_dir, mso, albedo)
        return render

    @cached
    def pbr_layered(self):
        return [
            self.pbr_layer(world_normal, self.view_dir(), mso, albedo)
            for world_normal, mso, albedo in zip(
                self.world_space_normal_layered(),
                self.mso_layered(),
                self.albedo_layered()
            )
        ]

    @cached
    def pbr(self):
        skybox = 0
        w = 0
        for light in self.scene.lights:
            if isinstance(light, ImageEnvironmentLight) and light.render_skybox:
                skybox = skybox + sample2d(light.image_rh(), float2(*unit_direction_to_latlong_uv(self.view_dir())))
                w += 1
        if w != 0:
            return self.compose_layers(
                self.pbr_layered() + [skybox / w],
                self.alpha_layered() + [torch.ones_like(skybox)]
            )
        else:
            return self.compose_layers(self.pbr_layered())
