import torch
from typing import Union, List
from .shader_ops import saturate, split_alpha, to_bchw, to_hwc


def background_alpha_compose(bg: Union[torch.Tensor, List[Union[float, int]], float, int], fgca: torch.Tensor):
    fgc, fga = split_alpha(fgca)
    if not torch.is_tensor(bg):
        bg = fgca.new_tensor(bg)
    return fgc * fga + bg * (1 - fga)


def alpha_blend(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc * (1 - fga) + fgc * fga, bga * (1 - fga) + fga


def additive(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc + fgc, saturate(bga + fga)


def alpha_additive(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc + fgc * fga, saturate(bga + fga)


def ssaa_downscale(rgb, factor: int):
    return to_hwc(torch.nn.functional.interpolate(to_bchw(rgb), scale_factor=1 / factor, mode='area'))
