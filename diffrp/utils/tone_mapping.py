import torch
import torch.nn.functional as F
from ..resources import get_resource_path
from .colors import linear_to_alexa_logc_ei1000, linear_to_srgb


def agx_base_contrast(rgb: torch.Tensor):
    lut: torch.Tensor = torch.load(get_resource_path("luts/agx-base-contrast.pt"))
    lut = lut.to(rgb)  # z y x 3
    logc = linear_to_alexa_logc_ei1000(rgb)  # h w 3
    # sampled: 1 c 1 h w
    return linear_to_srgb(F.grid_sample(
        lut[None].permute(0, 4, 1, 2, 3),
        logc[None, None] * 2 - 1,
        mode='bilinear',
        padding_mode='border'
    ).permute(0, 2, 3, 4, 1).reshape_as(rgb))
