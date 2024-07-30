import torch
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Optional, Dict, Union
from ..materials.base_material import SurfaceMaterial
from ..utils.shader_ops import zeros_like_vec, ones_like_vec, gpu_f32
from ..utils.geometry import make_face_soup, compute_face_normals, compute_vertex_normals


@dataclass
class MeshObject:
    # material
    material: SurfaceMaterial

    # geometry
    verts: torch.Tensor
    tris: torch.IntTensor
    normals: Union[Literal['flat', 'smooth'], torch.Tensor] = 'flat'

    # transform
    M: Optional[torch.Tensor] = None
    
    # attributes
    color: Optional[torch.Tensor] = None
    uv: Optional[torch.Tensor] = None
    tangents: Optional[torch.Tensor] = None
    custom_attrs: Optional[Dict[str, torch.Tensor]] = None

    def preprocess(self):
        if self.M is None:
            self.M = gpu_f32(torch.eye(4))
        if self.color is None:
            self.color = ones_like_vec(self.verts, 4)
        if self.uv is None:
            self.uv = zeros_like_vec(self.verts, 2)
        if self.tangents is None:
            self.tangents = zeros_like_vec(self.verts, 4)
        if self.custom_attrs is None:
            self.custom_attrs = {}
        if isinstance(self.normals, str):
            if self.normals == 'smooth':
                self.normals = compute_vertex_normals(self.verts, self.tris)
            if self.normals == 'flat':
                f = self.tris
                fn = compute_face_normals(self.verts, self.tris, normalize=True)
                self.verts, self.tris, self.normals = make_face_soup(self.verts, self.tris, fn)
                self.color = self.color[f].flatten(0, 1)
                self.uv = self.uv[f].flatten(0, 1)
                self.tangents = self.tangents[f].flatten(0, 1)
                new_customs = {}
                for k, attr in self.custom_attrs.items():
                    new_customs[k] = attr[f].flatten(0, 1)
                self.custom_attrs = new_customs
        return self
