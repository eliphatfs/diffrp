import torch
import nvdiffrast.torch as dr


class Interpolator:
    
    def __init__(self, vi_data: torch.Tensor, tris: torch.IntTensor):
        self.vi_data = vi_data  # (u, v, z|depth, tridx)
        self.tris = tris

    def interpolate(self, vertex_buffer: torch.Tensor):
        attr, da = dr.interpolate(vertex_buffer, self.vi_data, self.tris)
        return attr
