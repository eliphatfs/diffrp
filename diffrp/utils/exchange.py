import torch
import torch_redstone as rst
from PIL import Image
from .shader_ops import saturate


@torch.no_grad()
def to_pil(rgb_or_rgba: torch.Tensor):
    img = rst.torch_to_numpy((saturate(rgb_or_rgba.float()) * 255).byte().contiguous())
    return Image.fromarray(img)
