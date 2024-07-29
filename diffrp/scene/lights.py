import torch
from typing import Union
from dataclasses import dataclass


@dataclass
class Light(object):
    intensity: Union[torch.Tensor, float]


@dataclass
class DirectionalLight(Light):
    direction: torch.Tensor


@dataclass
class PointLight(Light):
    position: torch.Tensor


@dataclass
class ImageEnvironmentLight(Light):
    image: torch.Tensor
