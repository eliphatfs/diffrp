from typing import List
from .lights import Light
from .objects import MeshObject
from typing_extensions import Self


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
