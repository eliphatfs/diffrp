import torch
from .shader_ops import saturate, float4, split_31


def alpha_blend(background: torch.Tensor, foreground: torch.Tensor):
    rgb, a = split_31(foreground)
    return background * (1 - a) + float4(rgb * a, a)


def additive(background: torch.Tensor, foreground: torch.Tensor):
    brgb, ba = split_31(background)
    frgb, fa = split_31(foreground)
    return float4(brgb + frgb, saturate(ba + fa))
