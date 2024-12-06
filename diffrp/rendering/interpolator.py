"""
Implementation of different varying variable interpolators.

This is an implementation detail and has no guarantee on API stability.

See ``interpolator_impl`` in :py:class:`diffrp.rendering.surface_deferred.SurfaceDeferredRenderSessionOptions`
for explanation on selecting interpolators.
"""
import abc
import torch
from typing import Optional
import nvdiffrast.torch as dr
from ..utils import fma


def __float_as_int(x: torch.Tensor):
    return x.view(torch.int32)


def __int_as_float(x: torch.Tensor):
    return x.view(torch.float32)


def float_to_triidx(x: torch.Tensor):
    return torch.where(x <= 16777216, x.int(), __float_as_int(x) - 0x4a800000)


def triidx_to_float(x: torch.Tensor):
    return torch.where(x <= 0x01000000, x.float(), __int_as_float(0x4a800000 + x))


@torch.jit.script
def _interpolate_impl(
    vertex_buffer: torch.Tensor,
    vi_data: torch.Tensor,
    tris_idx: torch.Tensor,
    empty_region: Optional[float] = None,
):
    v123 = vertex_buffer[tris_idx]
    v1, v2, v3 = torch.select(v123, -2, 0), torch.select(v123, -2, 1), torch.select(v123, -2, 2)
    # v1: *, d
    u, v = vi_data[..., 0:1], vi_data[..., 1:2]
    # result = v1 * u + v2 * v + v3 * (1 - u - v)
    # (v1 - v3) * u + (v2 - v3) * v + v3
    result = fma(v1 - v3, u, fma(v2 - v3, v, v3))
    if empty_region is not None:
        result = torch.where(vi_data[..., -1:] > 0, result, empty_region)
    return result


def polyfill_interpolate(
    vertex_buffer: torch.Tensor,
    vi_data: torch.Tensor,
    tri_idx: torch.IntTensor,
    empty_region: Optional[float] = None,
):
    # n, d; f, 3; *, 4 -> *, d
    # tris[vi]: *, 3
    # v[tris[vi]]: *, 3, d
    # v1, v2, v3 = torch.unbind(vertex_buffer[tris[vi_idx - 1]], dim=-2)
    # v1: *, d
    # result = v1 * vi_data.x + v2 * vi_data.y + v3 * (1 - vi_data.x - vi_data.y)
    # if empty_region is not None:
    #     result = torch.where(vi_data.w > 0, result, empty_region)
    return _interpolate_impl(vertex_buffer, vi_data, tri_idx, empty_region)


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
        self.tri_idx = tris[float_to_triidx(self.vi_data[..., -1]) - 1].long()

    def interpolate(self, vertex_buffer: torch.Tensor):
        # n, d; f, 3; ?, 4
        return polyfill_interpolate(vertex_buffer, self.vi_data, self.tri_idx)
