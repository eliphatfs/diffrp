from .lights import Light
from .camera import Camera
from ..materials.base_material import VertexArrayObject


class Scene:
    def __init__(self) -> None:
        self.cameras = []
        self.lights = []
        self.objects = []

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def add_light(self, light: Light):
        self.lights.append(light)

    def add_object(self, obj):
        pass
