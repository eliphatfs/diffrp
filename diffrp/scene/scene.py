from typing import List
from .lights import Light
from .camera import Camera
from .objects import MeshObject


class Scene:
    def __init__(self) -> None:
        self.cameras: List[Camera] = []
        self.lights: List[Light] = []
        self.objects: List[MeshObject] = []

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def add_light(self, light: Light):
        self.lights.append(light)

    def add_mesh_object(self, mesh_obj: MeshObject):
        self.objects.append(mesh_obj.preprocess())
