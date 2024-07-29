import abc
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from collections.abc import Mapping
from ..utils.shader_ops import normalized
from ..utils.cache import cached, key_cached
from ..rendering.interpolator import Interpolator


@dataclass
class VertexArrayObject:
    # geometry
    verts: torch.Tensor
    tris: torch.IntTensor
    normals: torch.Tensor
    
    # attributes
    color: torch.Tensor
    uv: torch.Tensor
    tangents: torch.Tensor
    custom_attrs: Dict[str, torch.Tensor]


@dataclass
class SurfaceUniform:
    M: torch.Tensor
    V: torch.Tensor
    P: torch.Tensor
    custom_uniforms: Dict[str, torch.Tensor]

    @property
    @cached
    def camera_matrix(self):
        return torch.linalg.inv(self.V)

    @property
    @cached
    def camera_position(self):
        return self.camera_matrix[:3, 3]


class SurfaceInput:

    def __init__(self, uniforms: SurfaceUniform, vertex_buffers: VertexArrayObject, interpolator: Interpolator):
        self.cache = {}
        self.uniforms = uniforms
        self.interpolator = interpolator
        self.vertex_buffers = vertex_buffers
        self.custom_surface_inputs = CustomSurfaceInputs(self)

    def compute(self, vertex_buffer: torch.Tensor):
        return self.interpolator.interpolate(vertex_buffer)

    @property
    @cached
    def view_dir(self) -> torch.Tensor:
        # F3, view direction (normalized)
        # return self.compute(self.vertex_buffers)
        return normalized(self.world_pos - self.uniforms.camera_position)

    @property
    @cached
    def world_pos(self) -> torch.Tensor:
        # F3, world position
        return self.compute(self.vertex_buffers.verts)

    @property
    @cached
    def world_normal(self) -> torch.Tensor:
        # F3, geometry world normal (normalized)
        return normalized(self.compute(self.vertex_buffers.normals))

    @property
    @cached
    def color(self) -> torch.Tensor:
        # F4, vertex color, default to ones
        return self.compute(self.vertex_buffers.color)

    @property
    @cached
    def uv(self) -> torch.Tensor:
        # F2, uv, default to zeros
        return self.compute(self.vertex_buffers.uv)

    @property
    def custom_attrs(self) -> Mapping[str, torch.Tensor]:
        return self.custom_surface_inputs


class CustomSurfaceInputs(Mapping):

    def __init__(self, si: SurfaceInput) -> None:
        self.si = si

    @key_cached
    def __getitem__(self, attr: str) -> torch.Tensor:
        return self.si.compute(self.si.vertex_buffers.custom_attrs[attr])

    def __len__(self) -> int:
        return len(self.si.vertex_buffers.custom_attrs)

    def __iter__(self) -> iter:
        return iter(self.si.vertex_buffers.custom_attrs)


@dataclass
class SurfaceOutputStandard:
    albedo: Optional[torch.Tensor] = None  # F3, base color (diffuse or specular), default to magenta
    normal: Optional[torch.Tensor] = None  # F3, tangent space normal
    emission: Optional[torch.Tensor] = None  # F3, default to black
    metallic: Optional[torch.Tensor] = None  # F1, default to 0.0
    smoothness: Optional[torch.Tensor] = None  # F1, default to 0.5
    occlusion: Optional[torch.Tensor] = None  # F1, default to 1.0
    alpha: Optional[torch.Tensor] = None  # F1, default to 1.0
    aovs: Optional[Dict[str, torch.Tensor]] = None  # arbitrary, filled with nan for non-existing areas in render


class SurfaceMaterial(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        raise NotImplementedError
