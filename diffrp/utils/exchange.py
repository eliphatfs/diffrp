import torch
import torch_redstone as rst
from PIL import Image
from .shader_ops import saturate


@torch.no_grad()
def to_pil(rgb_or_rgba: torch.Tensor):
    """
    Convert a RGB or RGBA tensor to an PIL Image.
    The tensor is assumed to have range [0, 1].
    Values beyond that are clamped.

    Returns:
        Image: The converted PIL Image.
    """
    img = rst.torch_to_numpy((saturate(rgb_or_rgba.float()) * 255).byte().contiguous())
    return Image.fromarray(img)
