import torch
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Optional, Dict, Union, Any, TYPE_CHECKING
from ..utils.shader_ops import zeros_like_vec, ones_like_vec
from ..utils.geometry import make_face_soup, compute_face_normals, compute_vertex_normals
if TYPE_CHECKING:
    from ..materials.base_material import SurfaceMaterial


@dataclass
class MeshObject:
    """
    Represents a triangle mesh object in the scene.

    Args:
        material (SurfaceMaterial):
            The material to render with.
            You can use the :py:class:`diffrp.materials.default_material.DefaultMaterial`
            if you only care about the triangle geometry.
        verts (torch.Tensor):
            GPU tensor of vertices. Tensor of shape (V, 3), dtype float32.
        tris (torch.IntTensor):
            GPU tensor of indices. Tensor of shape (F, 3), dtype int32.
        normals (torch.Tensor or str):
            GPU tensor of vertex normals. Tensor of shape (V, 3), dtype float32.
            Alternatively, you can specify 'flat' or 'smooth' here.
            Automatic normals with flat or smooth faces will be computed.
            Defaults to 'flat'.
        M (torch.Tensor):
            Model matrix, or pose transform of the object.
            Tensor of shape (4, 4), dtype float32.
            Defaults to identity.
        colors (torch.Tensor):
            Linear-space RGBA vertex colors, in range [0, 1].
            Tensor of shape (V, 4), dtype float32.
            Defaults to white (all ones).
        uv (torch.Tensor):
            UV coordinates of vertices. Tensor of shape (V, 2), dtype float32.
            Defaults to zeros.
        tangents (torch.Tensor):
            Tangent spaces of vertices. Tensor of shape (V, 4), dtype float32.
            The first 3 dimensions are the tangent vector.
            The last dimension is sign of the bitangent.
            Defaults to zeros.
        custom_attrs (Dict[str, torch.Tensor]):
            Arbitrary attributes you want to bind to your vertices.
            Tensor of shape (V, \\*) for each attribute, dtype float32.
        metadata (Dict[str, Any]):
            Arbitrary meta data to tag the object. Not used for rendering.
    """
    # material
    material: 'SurfaceMaterial'

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
    
    metadata: Optional[Dict[str, Any]] = None

    def preprocess(self):
        if self.M is None:
            self.M = torch.eye(4, device=self.verts.device, dtype=self.verts.dtype)
        if self.color is None:
            self.color = ones_like_vec(self.verts, 4)
        if self.uv is None:
            self.uv = zeros_like_vec(self.verts, 2)
        if self.tangents is None:
            self.tangents = zeros_like_vec(self.verts, 4)
        if self.custom_attrs is None:
            self.custom_attrs = {}
        if self.metadata is None:
            self.metadata = {}
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
