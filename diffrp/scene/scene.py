from typing import List
from .lights import Light
from .objects import MeshObject


class Scene:
    def __init__(self) -> None:
        self.lights: List[Light] = []
        self.objects: List[MeshObject] = []

    def add_light(self, light: Light):
        self.lights.append(light)

    def add_mesh_object(self, mesh_obj: MeshObject):
        self.objects.append(mesh_obj.preprocess())
