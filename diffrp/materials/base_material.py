import abc
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from collections.abc import Mapping
from typing_extensions import Literal
from ..utils.cache import cached, key_cached
from ..rendering.interpolator import Interpolator
from ..utils.shader_ops import float4, normalized, transform_point4x3, transform_vector3x3, small_matrix_inverse


@dataclass
class VertexArrayObject:
    """
    Vertex Array Object. Includes vertex and index buffers of the scene.

    Created automatically by render pipelines.
    """
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
    """
    Users can access the system surface uniforms from this class.

    Uniform means the same value or data accessed across a material.
    Extra custom uniforms like textures and other attributes should be put in fields of the materials.

    Args:
        M (torch.Tensor): The Model matrix. Tensor of shape (4, 4).
        V (torch.Tensor): The View matrix in OpenGL convention. Tensor of shape (4, 4).
        P (torch.Tensor): The Projection matrix in OpenGL convention. Tensor of shape (4, 4).
    """
    M: torch.Tensor
    V: torch.Tensor
    P: torch.Tensor

    @property
    @cached
    def camera_matrix(self):
        """
        Access the camera matrix, or pose transform, in OpenGL convention.
        Camera right, up, forward is X, Y, -Z, respectively.
        This is the inverse of the View matrix ``V``.

        Returns:
            torch.Tensor: The camera matrix, commonly referred to as ``c2w``. Tensor of shape (4, 4).
        """
        return small_matrix_inverse(self.V)

    @property
    @cached
    def camera_position(self):
        """
        Access the position of the camera.

        Returns:
            torch.Tensor: The camera position vector XYZ. Tensor of shape (3,).
        """
        return self.camera_matrix[:3, 3]


class SurfaceInput:
    """
    Users can access system and custom attributes of the fragments currently processing
    from the ``shade`` function from this class.
    """

    def __init__(self, uniforms: SurfaceUniform, vertex_buffers: VertexArrayObject, interpolator: Interpolator):
        self.cache = {}
        self.uniforms = uniforms
        self.interpolator = interpolator
        self.vertex_buffers = vertex_buffers
        self.custom_surface_inputs = CustomSurfaceInputs(self)

    def interpolate_ex(
        self,
        vertex_buffer: torch.Tensor,
        world_transform: Literal['none', 'vector', 'vectornor', 'point', 'vector3ex1', 'vector3norex1'] = 'none',
    ):
        """
        Use this method to interpolate arbitrary vertex attributes into fragments queried in the ``shade`` method.

        Args:
            vertex_buffer (torch.Tensor): The vertex attributes to be interpolated. Tensor of shape (V, C).
            world_transform (str):
                | One of 'none', 'vector', 'vectornor', 'point', 'vector3ex1' or 'vector3norex1'.
                | **'none'**: Interpolate the vertex attributes as-is.
                | **'vector'**:
                    Requires ``C = 3``.
                    Transforms as direction vectors attributed with vertices into world space before interpolating.
                | **'vectornor'**:
                    Requires ``C = 3``.
                    Transforms as direction vectors attributed with vertices into world space
                    and normalize before interpolating.
                    They are **NOT** normalized again after interpolation.
                | **'point'**:
                    Requires ``C = 3``.
                    Transforms as positions or points attributed with vertices
                    into world space before interpolating.
                | **'vector3ex1'**:
                    Requires ``C = 4``.
                    Transforms the first 3 channels as direction vectors attributed with vertices
                    into world space before interpolating.
                    The fourth channel is interpolated as-is.
                | **'vector3norex1'**:
                    Requires ``C = 4``.
                    Transforms the first 3 channels as direction vectors attributed with vertices
                    into world space and normalize before interpolating.
                    They are **NOT** normalized again after interpolation.
                    The fourth channel is interpolated as-is.
        
        Returns:
            torch.Tensor:
                Tensor of shape (..., C) where ... is batched pixels.
                C is the number of channels in the vertex attribute you input in ``vertex_buffer``.
        """
        if world_transform == 'point':
            vertex_buffer = transform_point4x3(vertex_buffer, self.uniforms.M)
        elif world_transform == 'vector':
            vertex_buffer = transform_vector3x3(vertex_buffer, self.uniforms.M)
        elif world_transform == 'vectornor':
            vertex_buffer = normalized(transform_vector3x3(vertex_buffer, self.uniforms.M))
        elif world_transform == 'vector3ex1':
            vertex_buffer = float4(transform_vector3x3(vertex_buffer.xyz, self.uniforms.M), vertex_buffer.w)
        elif world_transform == 'vector3norex1':
            vertex_buffer = float4(normalized(transform_vector3x3(vertex_buffer.xyz, self.uniforms.M)), vertex_buffer.w)
        return self.interpolator.interpolate(vertex_buffer)

    @property
    @cached
    def view_dir(self) -> torch.Tensor:
        """
        View directions.
        The normalized unit vectors looking from the camera to the world position of the fragments.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 3) where ... is batched pixels.
        """
        return normalized(self.world_pos - self.uniforms.camera_position)

    @property
    @cached
    def world_pos(self) -> torch.Tensor:
        """
        The world position of the fragments.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 3) where ... is batched pixels.
        """
        return self.interpolate_ex(self.vertex_buffers.world_pos)

    @property
    @cached
    def local_pos(self) -> torch.Tensor:
        """
        The local (object space) position of the fragments.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 3) where ... is batched pixels.
        """
        return self.interpolate_ex(self.vertex_buffers.verts)

    @property
    @cached
    def world_normal_unnormalized(self) -> torch.Tensor:
        """
        The geometry world normal of the fragments.
        This includes the effect of smooth (vertex interpolation) and flat face specification in the ``MeshObject``.

        Normals are normalized in vertices, but not again after interpolation.
        Thus, this is slightly off from unit vectors, but may be useful in e.g. tangent space calculations.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 3) where ... is batched pixels.
        """
        return self.interpolate_ex(self.vertex_buffers.normals, 'vectornor')

    @property
    @cached
    def world_normal(self) -> torch.Tensor:
        """
        The normalized unit vectors for geometry world normal of the fragments.
        This includes the effect of smooth (vertex interpolation) and flat face specification in the ``MeshObject``.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 3) where ... is batched pixels.
        """
        return normalized(self.world_normal_unnormalized)

    @property
    @cached
    def color(self) -> torch.Tensor:
        """
        The interpolated color attributes (RGBA) of the fragments.
        Usually ranges from 0 to 1 (instead of 255).

        Defaults to white (ones) if not specified in the ``MeshObject``.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 4) where ... is batched pixels.
        """
        # F4, vertex color, default to ones
        return self.interpolate_ex(self.vertex_buffers.color)

    @property
    @cached
    def uv(self) -> torch.Tensor:
        """
        The interpolated UV coordinate attributes of the fragments.
        (0, 0) is the bottom-left in OpenGL convention, in contrast to top-left in most image libraries.

        Defaults to zeros if not specified in the ``MeshObject``.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 2) where ... is batched pixels.
        """
        return self.interpolate_ex(self.vertex_buffers.uv)

    @property
    @cached
    def world_tangent(self) -> torch.Tensor:
        """
        The interpolated tangents in world space of the fragments.

        The xyz dimensions form the tangent vector, while the w dimension specifies the sign (1 or -1).

        Defaults to zeros if not specified in the ``MeshObject``.

        Returns:
            torch.Tensor:
                Tensor of shape (..., 4) where ... is batched pixels.
        """
        return self.interpolate_ex(self.vertex_buffers.tangents, 'vector3norex1')

    @property
    def custom_attrs(self) -> Mapping[str, torch.Tensor]:
        """
        Dictionary of custom attributes of the fragments.

        The keys are the same as you specified in the ``MeshObject``.

        If some objects provide the custom inputs while others not for the key,
        you will get zeros for objects where the custom vertex attributes are absent.

        An error will be raised if custom inputs with the same key have different numbers of channels.

        Returns:
            torch.Tensor:
                Tensor of shape (..., C) where ... is batched pixels.
                C is the number of channels in the vertex attribute you input in ``MeshObject``.
        """
        return self.custom_surface_inputs


class CustomSurfaceInputs(Mapping):
    """
    Implementation for custom vertex attribute access.

    This is an implementation detail and you do not need to take care of it.
    """

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
    """
    The output structure that specifies standard and custom (AOV) outputs from the material (shader).
    All outputs are optional.
    The outputs shapes are all (..., C), where (...) is batched pixels.
    They should match the batch dimensions provided by ``SurfaceInput``.
    It can be 1D to 3D according to interpolator implementation, 1D under default settings
    -- a batch of pixel features of shape (B, C).
    See the documentation for interpolator options and interpolators for more details.
    
    Args:
        albedo (torch.Tensor):
            The per-pixel base color in linear RGB space, used both for diffuse and specular.
            Defaults to magenta (1.0, 0.0, 1.0).
            Tensor of shape (..., 3), where (...) should match the pixel batch dimensions in inputs.
        normal (torch.Tensor):
            The per-pixel normal in specified space. Defaults to geometry normal.
            See also ``normal_space``.
            This can be used to implement per-pixel (neural) normal maps.
            Tensor of shape (..., 3), where (...) should match the pixel batch dimensions in inputs.
        emission (torch.Tensor):
            The per-pixel emission value in linear RGB space.
            Defaults to black (0.0, 0.0, 0.0) which means no emission.
            Tensor of shape (..., 3), where (...) should match the pixel batch dimensions in inputs.
        metallic (torch.Tensor):
            The per-pixel metallic value.
            Defaults to 0.0, fully dielectric.
            Tensor of shape (..., 1), where (...) should match the pixel batch dimensions in inputs.
        smoothness (torch.Tensor):
            The per-pixel smoothness value.
            Defaults to 0.5.
            Tensor of shape (..., 1), where (...) should match the pixel batch dimensions in inputs.
        occlusion (torch.Tensor):
            The per-pixel ambient occlusion value. Dims indirect light.
            Defaults to 1.0, no occlusion.
            Tensor of shape (..., 1), where (...) should match the pixel batch dimensions in inputs.
        alpha (torch.Tensor):
            The per-pixel alpha value. Linear transparency.
            Defaults to 1.0, fully opaque.
            Any value will be overridden to 1.0 if render sessions are executed in ``opaque_only`` mode.
            Tensor of shape (..., 1), where (...) should match the pixel batch dimensions in inputs.
        aovs (Dict[str, torch.Tensor]):
            Auxiliary Output Variables (AOVs).
            Anything you would like to output for the shader.
            Batch dimensions of values need to match the pixel batch dimensions in inputs.
        normal_space(str):
            | One of 'tangent', 'object' and 'world'.
            | Defaults to 'tangent' as most normal maps are in tangent space,
              featuring a 'blueish' look.
            | 'object' and 'world' spaces are simple.
            | 'tangent' means (tangent, bitangent, geometry_normal) for XYZ components, respectively,
              where tangents are typically generated from the UV mapping.
              See also the ``mikktspace`` plugin.
    """
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
        """
        The interface for implementing materials.
        It takes a fragment batch specified as ``SurfaceUniform`` and ``SurfaceInput``.
        
        You shall not assume any order or shape of the fragment batch.
        In other words, the method should be trivial on the batch dimension, that is,
        the method should return batched outputs equivalent to
        concatenated outputs from the same batch of input but split into multiple sub-batches.

        Most functions implemented with attribute accesses, element/vector-wise operations
        and DiffRP utility functions naturally have this property.
        """
        raise NotImplementedError
