import torch
from typing import Union, List
from .shader_ops import saturate, split_alpha


def background_alpha_compose(bg: Union[torch.Tensor, List[Union[float, int]], float, int], fgca: torch.Tensor):
    fgc, fga = split_alpha(fgca)
    if not torch.is_tensor(bg):
        bg = fgca.new_tensor(bg)
    return fgc * fga + bg * (1 - fga)


def alpha_blend(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc * (1 - fga) + fgc * fga, bga * (1 - fga) + fga


def additive(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc + fgc, saturate(bga + fga)
