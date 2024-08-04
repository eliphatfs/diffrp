import torch
import pyexr
from . import get_resource_path
from ..utils.cache import singleton_cached


@singleton_cached
def newport_loft():
    """
    Returns:
        torch.Tensor:
            a CPU float32 tensor of shape (800, 1600, 3) representing the environment of the NewPort Loft scene.
    """
    with pyexr.open(get_resource_path("hdri_exrs/newport_loft.exr")) as fi:
        return torch.tensor(fi.get())
