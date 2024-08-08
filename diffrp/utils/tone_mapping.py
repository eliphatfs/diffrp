import torch
from .cache import key_cached
from .shader_ops import sample3d
from ..resources import get_resource_path
from .colors import linear_to_alexa_logc_ei1000, linear_to_srgb


class AgxLutLoader:
    """
    :meta private:
    """
    @key_cached
    def load(self, variant: str) -> torch.Tensor:
        return torch.fliplr(torch.load(get_resource_path("luts/agx-%s.pt" % variant)).cuda()).contiguous()


agx_lut_loader = AgxLutLoader()


def agx_base_contrast(rgb: torch.Tensor):
    """
    Performs tone-mapping from HDR linear RGB space into LDR sRGB space
    with the state-of-the-art AgX tone-mapper, the base contrast variant.

    Args:
        rgb (torch.Tensor): Tensor of shape (..., 3) containing linear RGB values.
    Returns:
        torch.Tensor: Tensor of the same shape as the input.
    """
    lut = agx_lut_loader.load("base-contrast")
    lut = lut.to(rgb)  # z y x 3
    logc = linear_to_alexa_logc_ei1000(rgb)  # h w 3
    # sampled: 1 c 1 h w
    return linear_to_srgb(sample3d(lut, logc))
