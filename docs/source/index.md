# Welcome to DiffRP⚙️!

**DiffRP** aims to provide an easy-to-use programming interface for rendering and differentiable rendering pipelines.

```{figure} assets/spheres-nvdraa-4xssaa.jpg
:scale: 40 %
:alt: Sample Render with DiffRP PBR Rasterization.

The glTF [MetalRoughSpheresNoTextures](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/MetalRoughSpheresNoTextures/README.md) scene rendered with DiffRP PBR rasterization.
```

&NewLine;

```{figure} assets/game-1024spp-denoised.jpg
:scale: 60 %
:alt: Sample Render with DiffRP Path Tracing.

The glTF [ABeautifulGame](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ABeautifulGame) scene rendered with DiffRP Path Tracing.
```

## Installation

The package can be installed by:

```bash
pip install git+https://github.com/eliphatfs/diffrp
```

```{note}
DiffRP depends on PyTorch (`torch`). The default version `pip` resolves to may not come with the `cuda` version you want. It is recommended to install [PyTorch](https://pytorch.org/get-started/locally/#start-locally) before you install DiffRP so you can choose the version you like.
```

```{note}
DiffRP rendering is based on CUDA GPUs. You can develop without one, but a CUDA GPU is required to run the code.
```

### Other Dependencies

If you use rasterization in DiffRP, you need to have the CUDA development kit set up as we use the `nvdiffrast` backend. See also [https://nvlabs.github.io/nvdiffrast/#installation](https://nvlabs.github.io/nvdiffrast/#installation).

If you want to use hardware ray tracing in DiffRP, you need to install `torchoptix` by `pip install torchoptix`. See also the hardware and driver requirements [https://github.com/eliphatfs/torchoptix?tab=readme-ov-file#requirements](https://github.com/eliphatfs/torchoptix?tab=readme-ov-file#requirements).

If you plan on using plugins in DiffRP (currently when you compute tangents), `gcc` is required in path. This is already fulfilled in most Linux and Mac distributions. For Windows I recommend the Strawberry Perl [(https://strawberryperl.com/)](https://strawberryperl.com/) distribution of `gcc`.

## Get Started

Rendering attributes of a mesh programmingly is incredibly simple with DiffRP compared to conventional render engines like Blender and Unity.

In this section, we will get started to render a simple procedural mesh.

First, let's import what we would need:

```python
import trimesh.creation
from diffrp.scene import Scene, MeshObject
from diffrp.materials import DefaultMaterial
from diffrp.utils import gpu_f32, gpu_i32, to_pil
from diffrp.rendering.camera import PerspectiveCamera
from diffrp.rendering.surface_deferred import SurfaceDeferredRenderSession
```

We now create a icosphere and render the camera space normal map, where RGB color values in $[0, 1]$ map linearly to $[-1, 1]$ in normal XYZ vectors.

The first run may take some time due to compiling and importing the `nvdiffrast` backend. Following calls would be fast.

```python
# create the mesh (cpu)
mesh = trimesh.creation.icosphere(radius=0.8)
# initialize the DiffRP scene
scene = Scene()
# register the mesh, load vertices and faces arrays to GPU
scene.add_mesh_object(MeshObject(DefaultMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
# default camera at [0, 0, 3.2] looking backwards
camera = PerspectiveCamera()
# create the SurfaceDeferredRenderSession, a deferred-rendering rasterization pipeline session
rp = SurfaceDeferredRenderSession(scene, camera)
# convert output tensor to PIL Image and save
to_pil(rp.false_color_camera_space_normal()).save("procedural-normals.png")
```

Only 6 lines of code and we are done! The output should look like this:

```{figure} assets/procedural-normals.png
:scale: 25 %
:alt: Camera space normals of a procedurally-created sphere.
```

The GPU buffers can be replaced by arbitrary tensors or parameters, and the process has been fully differentiable.

```{important}
The `SurfaceDeferredRenderSession` is a "session", which means you have to recreate it when you want to change the scene and/or the camera to avoid out-dated cached data.
```

```{note}
DiffRP makes heavy use of device-local caching.
DiffRP operations do not support multiple GPUs on a single process.
You will need to access the GPUs with a distributed paradigm, where each process only uses one GPU.
```

### Next Steps

It is recommended to go through the glTF rendering tutorial even if you do not need the functionality. It helps learning basic graphics concepts in DiffRP.

```{toctree}
:maxdepth: 1
gltf
concepts
pipeline_surface_deferred
writing_a_material
ptpbr
pipeline_comparison
```

## API Reference

```{note}
All documented public APIs from submodules are available at the `diffrp.*` level except for resources and extension plugins.
```

```{toctree}
:maxdepth: 3
generated/diffrp
```

## Update Notes

+ **0.2.3**: Make submodule contents available at top level. Make context sharing mechanism thread-safe by default. Fix `torch.load` warnings in newer torch versions.
+ **0.2.2**: Fixed backward compatibility due to `staticmethod`.
+ **0.2.1**: Add metadata functionality.
+ **0.2.0**: Implemented path tracing pipeline.
+ **0.1.5**: Fix normal and tangent attributes in static batching, refactoring to extract logic shared across rendering pipelines.
+ **0.1.4**: Faster GLTF loading, static batching support, minor GPU memory optimizations.
+ **0.1.3**: Fix instanced GLTF material loading (GPU memory explosion issue).
+ **0.1.2**: Fix numerical stability of coordinate transform backward.
+ **0.1.1**: Performance and documentation improvements.
+ **0.1.0**: A major rewrite of the whole package. Will try to provide backward compatibility of all documented public APIs from this version on.
+ **0.0.2**: Minor improvements.
+ **0.0.1**: First verion, fully experimental.
