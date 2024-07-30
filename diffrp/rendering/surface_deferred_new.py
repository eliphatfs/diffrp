import torch
import nvdiffrast.torch as dr
from typing import Union, List, Tuple, Callable
from ..scene.scene import Scene
from ..utils.cache import cached
from .interpolator import Interpolator
from ..utils.shader_ops import gpu_f32, transform_point, transform_point4x3, zeros_like_vec
from ..materials.base_material import VertexArrayObject, SurfaceMaterial, SurfaceInput, SurfaceUniform, SurfaceOutputStandard


class SurfaceDeferredRenderSession:
    def __init__(
        self,
        ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext],
        scene: Scene,
        max_layers: int = 1
    ) -> None:
        self.ctx = ctx
        self.scene = scene
        self.max_layers = max_layers

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
            torch.cat(tris).int().contiguous(),
            self._checked_cat([obj.normals for obj in objs], verts_ref, 3),
            torch.cat(world_pos),
            torch.cat(sts).short().contiguous(),
            self._checked_cat([obj.color for obj in objs], verts_ref, None),
            self._checked_cat([obj.uv for obj in objs], verts_ref, 2),
            self._checked_cat([obj.tangents for obj in objs], verts_ref, 4),
            {k: torch.cat([obj.custom_attrs[k] for obj in objs]) for k in total_custom_attrs}
        )
    
    @cached
    def rasterize(self):
        vao = self.vertex_array_object()
        rast_buffers: List[List[torch.Tensor]] = []
        for camera in self.scene.cameras:
            v = camera.V()
            p = camera.P()
            h, w = camera.resolution()
            layers = []
            clip_space = transform_point(vao.world_pos, p @ v).contiguous()
            with dr.DepthPeeler(self.ctx, clip_space, vao.tris.contiguous(), (h, w)) as dp:
                for i in range(self.max_layers):
                    rast, rast_db = dp.rasterize_next_layer()
                    if (i < self.max_layers - 1) and (rast.a <= 0).all():
                        break
                    layers.append(rast)
            rast_buffers.append(layers)
        return rast_buffers

    @cached
    def layer_material(self):
        results: List[List[List[Tuple[SurfaceInput, SurfaceOutputStandard]]]] = []
        vao = self.vertex_array_object()
        for cam, layers in zip(self.scene.cameras, self.rasterize()):
            layers_output = []
            for rast in layers:
                mats = []
                for x in self.scene.objects:
                    su = SurfaceUniform(x.M, cam.V(), cam.P())
                    si = SurfaceInput(su, vao, Interpolator(rast, vao.tris))
                    so = x.material.shade(su, si)
                    mats.append((si, so))
                layers_output.append(mats)
            results.append(layers_output)
        return results

    def gbuffer_collect(
        self,
        operator: Callable[[SurfaceInput, SurfaceOutputStandard], torch.Tensor],
        default: Union[List[float], torch.Tensor]
    ):
        for cam, layers in zip(self.scene.cameras, self.layer_material()):
            h, w = cam.resolution()
            for rast in layers:
                buffer = gpu_f32(default).expand(1, h, w, len(default))

    @cached
    def albedo(self):
        return self.gbuffer_collect(lambda x, y: y.albedo, [1.0, 0.0, 1.0])
