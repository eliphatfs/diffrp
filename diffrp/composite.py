import torch
from .shader_ops import saturate, float4


def alpha_blend(background: torch.Tensor, foreground: torch.Tensor):
    alpha = saturate(foreground.a)
    return background * (1 - alpha) + foreground * alpha


def additive(background: torch.Tensor, foreground: torch.Tensor):
    return float4(background.rgb + foreground.rgb, saturate(background.a + foreground.a))
