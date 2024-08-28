import torch
import collections
from typing import List
from .lights import Light
from .objects import MeshObject
from typing_extensions import Self
from ..utils import transform_point4x3


class Scene:
    """
    Represents a scene in DiffRP.
    ``add_*`` methods return the scene itself for chaining calls.
    """
    def __init__(self) -> None:
        self.lights: List[Light] = []
        self.objects: List[MeshObject] = []

    def add_light(self, light: Light) -> Self:
        self.lights.append(light)
        return self

    def add_mesh_object(self, mesh_obj: MeshObject) -> Self:
        self.objects.append(mesh_obj.preprocess())
        return self

    def static_batching(self) -> Self:
        """
        Batches meshes with the same material into a single pass for faster rendering.
        Suitable if you want to render multiple frames of the same scene.

        Changes are made in-place.
        You shall not modify the old mesh objects after calling this method.
        Local coordinates may change for individual meshes as they are combined.
        The world space and most renders will remain the same after the operation.
        
        Meshes with the same material are required to have the same set of custom attributes.

        Returns:
            Scene: The scene itself after in-place modification.
        """
        material_list = collections.defaultdict(list)
        for mesh in self.objects:
            material_list[id(mesh.material)].append(mesh)

        def _combine(meshes: List[MeshObject]) -> MeshObject:
            if len(meshes) == 1:
                return meshes[0]
            assert len(meshes) > 1
            tris = []
            offset = 0
            for x in meshes:
                tris.append(x.tris + offset)
                offset += len(x.verts)
            return MeshObject(
                x.material,
                torch.cat([transform_point4x3(x.verts, x.M) for x in meshes]),
                torch.cat(tris),
                torch.cat([x.normals for x in meshes]),
                torch.eye(4, dtype=torch.float32, device=x.verts.device),
                torch.cat([x.color for x in meshes]),
                torch.cat([x.uv for x in meshes]),
                torch.cat([x.tangents for x in meshes]),
                {k: torch.cat([x.custom_attrs[k] for x in meshes]) for k in x.custom_attrs}
            )

        self.objects = [_combine(x) for x in material_list.values()]
        return self
