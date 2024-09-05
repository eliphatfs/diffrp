"""
Experimental. Implementation and interface subject to change.
"""
import abc
import torch
from typing import Tuple
from typing_extensions import Literal

from .shader_ops import cross, dot, full_like_vec


class Raycaster(metaclass=abc.ABCMeta):
    def __init__(self, verts: torch.Tensor, tris: torch.IntTensor, config: dict = {}) -> None:
        self.config = config
        self.build(verts, tris, config)

    @abc.abstractmethod
    def build(self, verts: torch.Tensor, tris: torch.IntTensor, config: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, rays_o: torch.Tensor, rays_d: torch.Tensor, far: float) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BruteForceRaycaster(Raycaster):
    @staticmethod
    @torch.jit.script
    def _ray_tri_intersect(
        rays_o: torch.Tensor, rays_d: torch.Tensor, far: float, tris: torch.Tensor, epsilon: float
    ):
        # rays and tri verts should be broadcastable, and triangles should have at most one more dim
        # ..., 1/M, 3
        v1, v2, v0 = torch.unbind(tris, -2)  # M, 3 or ..., M, 3
        e1 = v1 - v0
        e2 = v2 - v0
        cr = cross(rays_d, e2)
        det = dot(e1, cr)
        inv_det = torch.reciprocal(det)
        s = rays_o - v0
        u = inv_det * dot(s, cr)
        scross1 = cross(s, e1)
        v = inv_det * dot(rays_d, scross1)
        t = inv_det * dot(e2, scross1)  # ..., M, 1
        hit_mask = (
            (torch.abs(det) > epsilon) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        )
        t = torch.where(hit_mask, t, far)  # ..., M, 1
        return t

    @torch.no_grad()
    def build(self, verts: torch.Tensor, tris: torch.IntTensor, config: dict):
        self.triangles = verts[tris]

    @torch.no_grad()
    def query(self, rays_o: torch.Tensor, rays_d: torch.Tensor, far: float):
        t = BruteForceRaycaster._ray_tri_intersect(
            rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), far, self.triangles.unsqueeze(-4),
            self.config.get("epsilon", 1e-6)
        )
        sel_tri = torch.argmin(t, dim=-2, keepdim=True)  # ..., M, 1 -> ..., 1, 1
        sel_t = torch.gather(t, -2, sel_tri).squeeze(-1)  # ..., 1
        fill_idx = sel_tri.squeeze(-1)
        return sel_t, fill_idx


@torch.jit.script
def _expand_bits(v: torch.Tensor):
    # long/int tensor
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


@torch.jit.script
def zorder_3d(xyz: torch.Tensor):
    xyz = xyz - xyz.min(0).values
    xyz = xyz / (xyz.max(0).values + 1e-8)
    xyz = (xyz * 1023).int()
    xyz = _expand_bits(xyz)
    x, y, z = torch.unbind(xyz, dim=-1)
    return x | (y << 1) | (z << 2)


class NaivePBBVH(Raycaster):

    @torch.no_grad()
    def build(self, verts: torch.Tensor, tris: torch.IntTensor, config: dict):
        M = len(tris)
        n = (M - 1).bit_length()

        triangles = verts[tris]

        builder: Literal['morton', 'splitaxis'] = config.get('builder', 'splitaxis')
        
        with torch.no_grad():
            if M != 2 ** n:
                triangles = torch.cat([triangles, triangles[:2 ** n - M]])
            centroids = triangles.mean(-2)
            if builder == 'morton':
                rank = torch.argsort(zorder_3d(centroids))
            elif builder == 'splitaxis':
                rank = torch.arange(len(triangles), device=verts.device)[None]  # 1, M
                smin = triangles.min(-2).values[None]  # 1, ^M^, 3
                smax = triangles.max(-2).values[None]
                centroids = centroids[None]
                bmins = []
                bmaxs = []
                # 2 ** i, m, 3
                while smin.shape[-2] > 1:
                    sbmin = smin.min(-2).values  # 2 ** i, 3
                    sbmax = smax.max(-2).values  # 2 ** i, 3
                    axes = torch.argmax(sbmax - sbmin, dim=-1)  # 2 ** i
                    i2, m = rank.shape
                    b = torch.arange(i2, device=centroids.device)
                    arg = torch.argsort(centroids[b, :, axes], dim=-1)  # 2 ** i, m
                    b = b.unsqueeze(-1)
                    smin = smin[b, arg].view(2 * i2, m // 2, 3)
                    smax = smax[b, arg].view(2 * i2, m // 2, 3)
                    centroids = centroids[b, arg].view(2 * i2, m // 2, 3)
                    rank = rank[b, arg].view(2 * i2, m // 2)
                rank = rank.view(2 ** n)
            elif builder == 'none':
                rank = torch.arange(len(triangles), device=verts.device)
            else:
                assert False, 'Unsupported builder: %s' % builder
            
            skip_next = torch.arange(2, (1 << (n + 1)) + 1, device=verts.device)
            skip_next = skip_next // ((-skip_next) & skip_next) - 1

            scan_next = torch.arange(1, (1 << (n + 2)) - 2, 2, device=verts.device)
            scan_next = torch.where(scan_next >= (1 << (n + 1)) - 1, skip_next, scan_next)

            bmins = [triangles.min(-2).values[rank]]
            bmaxs = [triangles.max(-2).values[rank]]
            while len(bmins[-1]) > 1:
                bmins.append(torch.minimum(*bmins[-1].view(-1, 2, 3).unbind(-2)))
                bmaxs.append(torch.maximum(*bmaxs[-1].view(-1, 2, 3).unbind(-2)))
            bmins.reverse()
            bmaxs.reverse()

            bmins = torch.cat(bmins)
            bmaxs = torch.cat(bmaxs)

        self.M = M
        self.n = n
        self.rank = rank % M
        self.bmins = bmins
        self.bmaxs = bmaxs
        self.skip_next = skip_next
        self.scan_next = scan_next
        self.triangles = triangles[rank]

    @staticmethod
    @torch.jit.script
    def _ray_box_intersect_impl(
        bmins: torch.Tensor, bmaxs: torch.Tensor, traverser: torch.Tensor,
        rays_o_live: torch.Tensor, rays_d_live: torch.Tensor, t_live: torch.Tensor,
        tri_start: int
    ):
        bound_min = bmins[traverser]
        bound_max = bmaxs[traverser]
        t1 = (bound_min - rays_o_live) / rays_d_live  # R, 3
        t2 = (bound_max - rays_o_live) / rays_d_live  # R, 3
        t_min = torch.min(t1, t2).max(-1).values
        t_max = torch.max(t1, t2).min(-1).values
        intersect_mask = (t_min <= t_live) & (t_max > 0) & (t_min <= t_max)
        return intersect_mask, intersect_mask & (traverser >= tri_start)

    @torch.no_grad()
    def query(self, rays_o: torch.Tensor, rays_d: torch.Tensor, far: float):
        epsilon = self.config.get("epsilon", 1e-6)
        traverser = rays_o.new_zeros(rays_o.shape[:-1], dtype=torch.int64)  # R
        # ^N^, 3
        # R, 3
        live_idx = torch.arange(len(traverser), device=traverser.device)
        t = full_like_vec(rays_o, far, 1)  # R, 3
        i = torch.zeros([len(rays_o), 1], device=traverser.device, dtype=torch.int64)
        tri_start = (1 << self.n) - 1
        while True:
            live_ray = traverser >= 0
            t_live = t[live_idx].squeeze(-1)
            rays_o_live = rays_o[live_idx]
            rays_d_live = rays_d[live_idx]
            for _ in range((self.n + 1) // 2):
                intersect_mask, tri_candidate = NaivePBBVH._ray_box_intersect_impl(
                    self.bmins, self.bmaxs, traverser, rays_o_live, rays_d_live, t_live, tri_start
                )

                tri_hits = tri_candidate.nonzero(as_tuple=True)
                if len(tri_hits[0]) > 0:
                    ray_idx = live_idx[tri_hits]
                    tri_idx = traverser[tri_hits] - tri_start
                    test_t = BruteForceRaycaster._ray_tri_intersect(
                        rays_o[ray_idx], rays_d[ray_idx], far,
                        self.triangles[tri_idx], epsilon
                    )
                    t.scatter_reduce_(0, ray_idx.unsqueeze(-1), test_t, 'amin')
                    i[ray_idx] = torch.where(test_t <= t[ray_idx], tri_idx.unsqueeze(-1), i[ray_idx])
                    del test_t, tri_idx, ray_idx
                del tri_hits
                traverser = torch.where(intersect_mask, self.scan_next[traverser], self.skip_next[traverser])
                live_ray &= traverser != 0
            live_ray = live_ray.nonzero(as_tuple=True)
            if len(live_ray[0]) == 0:
                break
            traverser = traverser[live_ray]
            live_idx = live_idx[live_ray]
            del live_ray
        return t, self.rank[i]
