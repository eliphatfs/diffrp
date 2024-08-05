"""
Implementation of different varying variable interpolators.

This is an implementation detail and has no guarantee on API stability.

See ``intepolator_impl`` in :py:class:`diffrp.rendering.surface_deferred.SurfaceDeferredRenderSessionOptions`
for explanation on selecting interpolators.
"""
import abc
import torch
from typing import Optional
import nvdiffrast.torch as dr


def polyfill_interpolate(
    vertex_buffer: torch.Tensor,
    vi_data: torch.Tensor,
    tris: torch.IntTensor,
    empty_region: Optional[float] = None
):
    # n, d; f, 3; *, 4 -> *, d
    # tris[vi]: *, 3
    # v[tris[vi]]: *, 3, d
    v1, v2, v3 = torch.unbind(vertex_buffer[tris[vi_data.w.int().squeeze(-1) - 1]], dim=-2)
    # v1: *, d
    result = v1 * vi_data.x + v2 * vi_data.y + v3 * (1 - vi_data.x - vi_data.y)
    if empty_region is not None:
        result = torch.where(vi_data.w > 0, result, empty_region)
    return result


class Interpolator(metaclass=abc.ABCMeta):
    vi_data: torch.Tensor

    @abc.abstractmethod
    def interpolate(self, vertex_buffer: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FullScreenInterpolator(Interpolator):
    
    def __init__(self, vi_data: torch.Tensor, tris: torch.IntTensor):
        self.vi_data = vi_data.contiguous()  # (u, v, z|depth, tridx)
        self.tris = tris.contiguous()

    def interpolate(self, vertex_buffer: torch.Tensor):
        attr, da = dr.interpolate(vertex_buffer.contiguous(), self.vi_data, self.tris)
        return attr


class MaskedSparseInterpolator(Interpolator):
    
    def __init__(self, vi_data: torch.Tensor, tris: torch.IntTensor, mask: torch.BoolTensor):
        self.tris = tris
        self.indices = mask.nonzero(as_tuple=True)
        self.vi_data = vi_data[self.indices]  # (u, v, z|depth, tridx)

    def interpolate(self, vertex_buffer: torch.Tensor):
        # n, d; f, 3; ?, 4
        return polyfill_interpolate(vertex_buffer, self.vi_data, self.tris)
