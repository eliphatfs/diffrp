import torch
from dataclasses import dataclass
from typing_extensions import Literal
from typing import Optional, Dict, Union
from ..materials.base_material import SurfaceMaterial


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
