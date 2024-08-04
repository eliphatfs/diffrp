import torch
from typing import Optional
from .base_material import SurfaceInput, SurfaceMaterial, SurfaceUniform, SurfaceOutputStandard


class DefaultMaterial(SurfaceMaterial):
    def __init__(self, tint: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.tint = tint

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        if self.tint is not None:
            return SurfaceOutputStandard(si.color.rgb * self.tint)
        return SurfaceOutputStandard(si.color.rgb)
