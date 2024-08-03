import torch
from typing import Union
from dataclasses import dataclass


@dataclass
class Light(object):
    intensity: Union[torch.Tensor, float]
    color: torch.Tensor


@dataclass
class DirectionalLight(Light):
    direction: torch.Tensor


@dataclass
class PointLight(Light):
    position: torch.Tensor


@dataclass
class ImageEnvironmentLight(Light):
    image: torch.Tensor
    render_skybox: bool = True

    def image_rh(self):
        return torch.fliplr(self.image) * (self.intensity * self.color)
