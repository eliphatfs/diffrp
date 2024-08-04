import torch
from typing import Optional
from .base_material import SurfaceInput, SurfaceMaterial, SurfaceUniform, SurfaceOutputStandard


class DefaultMaterial(SurfaceMaterial):
    """
    Default material.

    Outputs the interpolated vertex color attributes combined with an optional tint as the albedo,
    and other attributes remain the default.

    Args:
        tint(Optional[torch.Tensor]): Linear multipliers for colors. Defaults to ``None``.
    """

    def __init__(self, tint: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.tint = tint

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        if self.tint is not None:
            return SurfaceOutputStandard(si.color.rgb * self.tint)
        return SurfaceOutputStandard(si.color.rgb)
