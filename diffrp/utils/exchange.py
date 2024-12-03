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


def torch_load_no_warning(fn: str):
    if torch.__version__ >= '2.4':
        return torch.load(fn, weights_only=True)
    else:
        return torch.load(fn)
