import torch
from typing import Union, List
from .shader_ops import saturate, split_alpha, to_bchw, to_hwc, fma


def background_alpha_compose(bg: Union[torch.Tensor, List[Union[float, int]], float, int], fgca: torch.Tensor):
    """
    Compose a background color to a foreground via alpha blending.

    It not only applies to RGB color and RGBA foregrounds, but also general linear data.

    The background should be one element shorter than the foreground elements.

    Args:
        bg: Tensor or List[float] or number.
            A single number is equivalent to the RGB list filled by it.
            The length should be one less than foreground.
        fgca: Tensor with alpha, shape (..., C + 1).
    Returns:
        torch.Tensor: Composed result, shape (..., C).
    """
    fgc, fga = split_alpha(fgca)
    if not isinstance(bg, (float, int)) and not isinstance(bg, torch.Tensor):
        bg = fgca.new_tensor(bg)
    return fgc * fga + bg * (1 - fga)


def alpha_blend(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return torch.lerp(bgc, fgc, fga), fma(bga, 1 - fga, fga)


def additive(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc + fgc, saturate(bga + fga)


def alpha_additive(bgc: torch.Tensor, bga: torch.Tensor, fgc: torch.Tensor, fga: torch.Tensor):
    return bgc + fgc * fga, saturate(bga + fga)


def ssaa_downscale(rgb, factor: int):
    return to_hwc(torch.nn.functional.interpolate(to_bchw(rgb), scale_factor=1 / factor, mode='area'))
