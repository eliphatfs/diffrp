import torch
from dataclasses import dataclass
from typing_extensions import Literal
from ..render_pipelines.surface_deferred import SurfaceInput, SurfaceMaterial, SurfaceOutputStandard, SurfaceUniform
from ..shader_ops import sample2d, ones_like_vec


@dataclass
class GLTFSampler:
    image: torch.Tensor  # F3 or F4
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

    normal_texture: GLTFSampler
    occlusion_texture: GLTFSampler

    emissive_factor: torch.Tensor  # F3
    emissive_texture: GLTFSampler

    alpha_cutoff: float
    alpha_mode: Literal['OPAQUE', 'MASK', 'BLEND']

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        rgba = self.base_color_factor * si.color * self.base_color_texture.sample(si.uv)
        mr = self.metallic_roughness_texture.sample(si.uv)

        if self.alpha_mode == 'OPAQUE':
            alpha = ones_like_vec(rgba, 1)
        elif self.alpha_mode == 'MASK':
            alpha = (rgba.a > self.alpha_mode).float()
        elif self.alpha_mode == 'BLEND':
            alpha = rgba.a
        else:
            raise ValueError('bad GLTFMaterial.alpha_mode', self.alpha_mode)

        return SurfaceOutputStandard(
            rgba.rgb,
            self.normal_texture.sample(si.uv) * 2 + 1,
            self.emissive_factor * self.emissive_texture.sample(si.uv),
            self.metallic_factor * mr.b,
            1.0 - self.roughness_factor * mr.g,
            self.occlusion_texture.sample(si.uv).r,
            alpha
        )
