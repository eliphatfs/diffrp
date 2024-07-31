import abc
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from collections.abc import Mapping
from typing_extensions import Literal
from ..utils.cache import cached, key_cached
from ..rendering.interpolator import Interpolator
from ..utils.shader_ops import float4, normalized, transform_point4x3, transform_vector3x3


@dataclass
class VertexArrayObject:
    # geometry
    verts: torch.Tensor
    normals: torch.Tensor
    world_pos: torch.Tensor

    # index buffers
    tris: torch.IntTensor
    stencils: torch.IntTensor
    # REVIEW: this seems coupled with implementation details of render pipelines
    # can we make this only for the material?
    
    # vertex attributes
    color: torch.Tensor
    uv: torch.Tensor
    tangents: torch.Tensor
    custom_attrs: Dict[str, torch.Tensor]


@dataclass
class SurfaceUniform:
    M: torch.Tensor
    V: torch.Tensor
    P: torch.Tensor

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

    def interpolate_ex(
        self,
        vertex_buffer: Optional[torch.Tensor],
        world_transform: Literal['none', 'vector', 'point', 'vector3ex1'] = 'none'
    ):
        if world_transform == 'point':
            vertex_buffer = transform_point4x3(vertex_buffer, self.uniforms.M)
        elif world_transform == 'vector':
            vertex_buffer = transform_vector3x3(vertex_buffer, self.uniforms.M)
        elif world_transform == 'vector3ex1':
            vertex_buffer = float4(transform_vector3x3(vertex_buffer.xyz, self.uniforms.M), vertex_buffer.w)
        return self.interpolator.interpolate(vertex_buffer)

    @property
    @cached
    def view_dir(self) -> torch.Tensor:
        # F3, view direction (normalized)
        return normalized(self.world_pos - self.uniforms.camera_position)

    @property
    @cached
    def world_pos(self) -> torch.Tensor:
        # F3, world position
        return self.interpolate_ex(self.vertex_buffers.world_pos)

    @property
    @cached
    def local_pos(self) -> torch.Tensor:
        # F3, world position
        return self.interpolate_ex(self.vertex_buffers.verts)

    @property
    @cached
    def world_normal(self) -> torch.Tensor:
        # F3, geometry world normal (normalized)
        return normalized(self.interpolate_ex(self.vertex_buffers.normals, 'vector'))

    @property
    @cached
    def color(self) -> torch.Tensor:
        # F4, vertex color, default to ones
        return self.interpolate_ex(self.vertex_buffers.color)

    @property
    @cached
    def uv(self) -> torch.Tensor:
        # F2, uv, default to zeros
        return self.interpolate_ex(self.vertex_buffers.uv)

    @property
    @cached
    def world_tangent(self) -> torch.Tensor:
        # F2, uv, default to zeros
        return self.interpolate_ex(self.vertex_buffers.tangents, 'vector3ex1')

    @property
    def custom_attrs(self) -> Mapping[str, torch.Tensor]:
        return self.custom_surface_inputs


class CustomSurfaceInputs(Mapping):

    def __init__(self, si: SurfaceInput) -> None:
        self.si = si

    @key_cached
    def __getitem__(self, attr: str) -> torch.Tensor:
        return self.si.interpolate_ex(self.si.vertex_buffers.custom_attrs[attr])

    def __len__(self) -> int:
        return len(self.si.vertex_buffers.custom_attrs)

    def __iter__(self) -> iter:
        return iter(self.si.vertex_buffers.custom_attrs)


@dataclass
class SurfaceOutputStandard:
    albedo: Optional[torch.Tensor] = None  # F3, base color (diffuse or specular), default to magenta
    normal: Optional[torch.Tensor] = None  # F3, normal in specified space, default to geometry normal
    emission: Optional[torch.Tensor] = None  # F3, default to black
    metallic: Optional[torch.Tensor] = None  # F1, default to 0.0
    smoothness: Optional[torch.Tensor] = None  # F1, default to 0.5
    occlusion: Optional[torch.Tensor] = None  # F1, default to 1.0
    alpha: Optional[torch.Tensor] = None  # F1, default to 1.0
    aovs: Optional[Dict[str, torch.Tensor]] = None  # arbitrary, default to 0.0 for non-existing areas in render
    normal_space: Literal['tangent', 'object', 'world'] = 'tangent'


class SurfaceMaterial(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        raise NotImplementedError
