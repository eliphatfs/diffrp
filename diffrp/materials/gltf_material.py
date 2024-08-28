import torch
from typing import Optional
from dataclasses import dataclass
from typing_extensions import Literal
from ..utils.shader_ops import sample2d, fsa, ones_like_vec
from .base_material import SurfaceInput, SurfaceMaterial, SurfaceOutputStandard, SurfaceUniform


@dataclass
class GLTFSampler:
    image: torch.Tensor  # F1, F3 or F4
    wrap_mode: Literal['repeat', 'clamp', 'mirror'] = 'repeat'
    interpolation: Literal['point', 'linear'] = 'linear'

    def sample(self, uv: torch.Tensor):
        if self.wrap_mode == 'repeat':
            uv = uv.remainder(1.0)
        return sample2d(
            self.image, uv,
            wrap='border' if self.wrap_mode == 'clamp' else 'reflection',
            mode='bilinear' if self.interpolation == 'linear' else 'nearest'
        )


@dataclass
class GLTFMaterial(SurfaceMaterial):
    """
    A standard PBR material coherent with GLTF 2.0 Specifications.
    """
    base_color_factor: torch.Tensor  # F4
    base_color_texture: GLTFSampler

    metallic_factor: float
    roughness_factor: float
    metallic_roughness_texture: GLTFSampler

    normal_texture: Optional[GLTFSampler]
    occlusion_texture: Optional[GLTFSampler]

    emissive_factor: Optional[torch.Tensor]  # F3
    emissive_texture: GLTFSampler

    alpha_cutoff: float
    alpha_mode: Literal['OPAQUE', 'MASK', 'BLEND']

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        rgba = self.base_color_factor * si.color * self.base_color_texture.sample(si.uv)
        mr = self.metallic_roughness_texture.sample(si.uv)

        if self.alpha_mode == 'OPAQUE':
            alpha = None
        elif self.alpha_mode == 'MASK':
            alpha = (rgba.a > self.alpha_cutoff).float()
        elif self.alpha_mode == 'BLEND':
            alpha = rgba.a
        else:
            raise ValueError('bad GLTFMaterial.alpha_mode', self.alpha_mode)

        return SurfaceOutputStandard(
            rgba.rgb,
            fsa(2, self.normal_texture.sample(si.uv), -1) if self.normal_texture is not None else None,
            self.emissive_factor * self.emissive_texture.sample(si.uv) if self.emissive_factor is not None else None,
            self.metallic_factor * mr.b,
            fsa(-self.roughness_factor, mr.g, 1.0),
            self.occlusion_texture.sample(si.uv).r if self.occlusion_texture is not None else ones_like_vec(si.uv, 1),
            alpha
        )
