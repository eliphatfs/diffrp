import torch
from typing import Union
from dataclasses import dataclass


@dataclass
class Light(object):
    """
    Base class for Lights.

    Args:
        intensity (float): Linear multiplier of light intensity.
        color (torch.Tensor): Shape (3,). Linear RGB tint of light.
    """
    intensity: Union[torch.Tensor, float]
    color: torch.Tensor


@dataclass
class DirectionalLight(Light):
    """
    Directional light data structure; Not supported yet.
    """
    direction: torch.Tensor


@dataclass
class PointLight(Light):
    """
    Point light data structure; Not supported yet.
    """
    position: torch.Tensor


@dataclass
class ImageEnvironmentLight(Light):
    """
    Environment light based on lat-long format images.

    Args:
        intensity (float): Linear multiplier of light intensity.
        color (torch.Tensor): Shape (3,). Linear RGB tint of light.
        image (torch.Tensor): (H, 2H, 3) RGB linear values of lighting environment.
        render_skybox (bool): Whether to show this environment in PBR as the background.
    """
    image: torch.Tensor
    render_skybox: bool = True

    def image_rh(self):
        return torch.fliplr(self.image) * (self.intensity * self.color)
