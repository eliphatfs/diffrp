import torch
from ..scene import Scene
from .camera import Camera
from ..utils.cache import cached
from ..utils.shader_ops import *
from typing import List, Callable, Union, Tuple
from .interpolator import MaskedSparseInterpolator
from ..utils.coordinates import near_plane_ndc_grid
from ..materials.base_material import VertexArrayObject, SurfaceInput, SurfaceOutputStandard


class RenderSessionMixin:
    """
    Mixin implementation for common operations across render sessions.
    """
    scene: Scene
    camera: Camera
    
    @cached
    def camera_VP(self):
        return torch.mm(self.camera_P(), self.camera_V())
    
    @cached
    def camera_P(self):
        return self.camera.P()
    
    @cached
    def camera_V(self):
        return self.camera.V()
    
    @staticmethod
    def _view_dir_impl(grid: torch.Tensor, v: torch.Tensor, p: torch.Tensor, vp: torch.Tensor):
        inved_matrices = small_matrix_inverse(torch.stack([v, vp]))
        camera_pos = inved_matrices[0, :3, 3]
        grid = torch.matmul(grid, inved_matrices[1].T)
        grid = grid[..., :3] / grid[..., 3:]
        rays_d = normalized(grid - camera_pos)
        near = p[2, 3] / (p[2, 2] - 1)
        return camera_pos + rays_d * near, rays_d

    @cached
    def camera_far(self) -> float:
        p = self.camera_P()
        return (p[2, 3] / (p[2, 2] + 1)).item()

    @cached
    def camera_rays(self) -> torch.Tensor:
        grid = near_plane_ndc_grid(*self.camera.resolution(), torch.float32, torch.device('cuda'))
        return self._view_dir_impl(grid, self.camera_V(), self.camera_P(), self.camera_VP())

    @cached
    def view_dir(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor:
                Full-screen unit direction vectors of outbound rays from the camera.
                Tensor of shape (H, W, 3).
        """
        return self.camera_rays()[1]

    def _cat_fast_path(self, attrs: List[torch.Tensor]):
        if len(attrs) == 1:
            return attrs[0]
        else:
            return torch.cat(attrs)

    def _checked_cat(self, attrs: List[torch.Tensor], verts_ref: List[torch.Tensor], expected_dim):
        for v, a in zip(verts_ref, attrs):
            assert v.shape[0] == a.shape[0], "attribute length not the same as number of vertices"
            if isinstance(expected_dim, int):
                assert a.shape[-1] == expected_dim, "expected %d dims but got %d for vertex attribute" % (expected_dim, a.shape[-1])
        return self._cat_fast_path(attrs)

    @cached
    def vertex_array_object(self) -> VertexArrayObject:
        world_pos = []
        tris = []
        sts = [torch.full([1], 0, device='cuda', dtype=torch.int64)]
        objs = self.scene.objects
        offset = 0
        for s, x in enumerate(objs):
            world_pos.append(transform_point4x3(x.verts, x.M))
            assert x.tris.shape[-1] == 3, "Expected 3 vertices per triangle, got %d" % x.tris.shape[-1]
            tris.append(x.tris + offset)
            sts.append(torch.full([len(x.tris)], s + 1, dtype=torch.int64, device=x.tris.device))
            offset += len(x.verts)
        total_custom_attrs = set().union(*(obj.custom_attrs.keys() for obj in objs))
        for k in total_custom_attrs:
            for obj in objs:
                if k in obj.custom_attrs:
                    sz = obj.custom_attrs[k].shape[-1]
                    break
            else:
                assert False, "Internal assertion failure. You should never reach here."
            for obj in objs:
                if k in obj.custom_attrs:
                    assert len(obj.custom_attrs[k]) == len(obj.verts), "Attribute length not the same as number of vertices: %s" % k
                    assert obj.custom_attrs[k].shape[-1] == sz, "Different dimensions for same custom attribute: %d != %d" % (obj.custom_attrs[k].shape[-1], sz)
                else:
                    obj.custom_attrs[k] = zeros_like_vec(obj.verts, sz)

        verts_ref = [obj.verts for obj in objs]
        return VertexArrayObject(
            self._checked_cat([obj.verts for obj in objs], verts_ref, 3),
            self._checked_cat([obj.normals for obj in objs], verts_ref, 3),
            self._cat_fast_path(world_pos),
            self._cat_fast_path(tris).int(memory_format=torch.contiguous_format),
            self._cat_fast_path(sts).int(memory_format=torch.contiguous_format),
            self._checked_cat([obj.color for obj in objs], verts_ref, None),
            self._checked_cat([obj.uv for obj in objs], verts_ref, 2),
            self._checked_cat([obj.tangents for obj in objs], verts_ref, 4),
            {k: torch.cat([obj.custom_attrs[k] for obj in objs]) for k in total_custom_attrs}
        )

    def _collector_world_normal(self, si: SurfaceInput, so: SurfaceOutputStandard):
        if so.normal is None:
            return si.world_normal
        if so.normal_space == 'tangent':
            vn = si.world_normal_unnormalized
            vnt = so.normal
            vt, vs = si.world_tangent.xyz, si.world_tangent.w
            vb = vs * cross(vn, vt)
            return normalized(fma(vnt.x, vt, fma(vnt.y, vb, vnt.z * vn)))
        if so.normal_space == 'object':
            return normalized(transform_vector3x3(so.normal, si.uniforms.M))
        if so.normal_space == 'world':
            return normalized(so.normal)
        assert False, "Unknown normal space: " + so.normal_space

    def _gbuffer_collect_layer_impl_stencil_masked(
        self,
        mats: List[Tuple[SurfaceInput, SurfaceOutputStandard]],
        operator: Callable[[SurfaceInput, SurfaceOutputStandard], torch.Tensor],
        initial_buffer: torch.Tensor
    ):
        buffer = initial_buffer
        indices = []
        values = []
        nind = 0
        for si, so in mats:
            interpolator: MaskedSparseInterpolator = si.interpolator
            if len(si.interpolator.indices[0]) == 0:
                continue
            op = operator(si, so)
            if op is not None:
                nind = len(interpolator.indices)
                indices.append(interpolator.indices)
                values.append(op)
        if len(indices) == 1:
            buffer = buffer.index_put(indices[0], values[0])
        elif nind != 0:
            indices = tuple(torch.cat([x[i] for x in indices]) for i in range(nind))
            values = torch.cat(values)
            buffer = buffer.index_put(indices, values)
        return buffer
