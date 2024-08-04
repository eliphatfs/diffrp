import torch
import pyexr
from . import get_resource_path
from ..utils.cache import singleton_cached


@singleton_cached
def newport_loft():
    with pyexr.open(get_resource_path("hdri_exrs/newport_loft.exr")) as fi:
        return torch.tensor(fi.get())
