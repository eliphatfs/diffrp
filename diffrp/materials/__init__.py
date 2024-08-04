"""
Includes the interface and built-in implementations of materials.
The APIs in the subpackages are also directly available from this `diffrp.materials` package.
"""


from .base_material import SurfaceInput, SurfaceUniform, SurfaceOutputStandard, SurfaceMaterial
from .default_material import DefaultMaterial
from .gltf_material import GLTFSampler, GLTFMaterial
