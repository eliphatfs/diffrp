# Writing a Material

Most rendering requirements can be fulfilled by only coding the material by implementing a {py:class}`diffrp.materials.base_material.SurfaceMaterial`.

DiffRP is designed with modularity and extendability in mind. All interfaces are in the simple form of PyTorch function calls. For the default interpolator implementations, tensors you see in `SurfaceInput` will have shape (B, C) where B is batch number of pixels and C is the relevant channels, for example, positions and normals have 3 channels while RGBA colors have 4.

We show two examples: procedural effects and neural networks.

## Procedural Effects

We implement a material where the RGB is defined as functions of the world space coordinates.
For points with `z > 0.5`, we additionally increase the values by a star-like shape.

Note that within DiffRP you can use the GLSL-like attributes to access slices in Tensors.

```python
import torch
from diffrp.utils import float3
from diffrp.materials import SurfaceMaterial, SurfaceInput, SurfaceOutputStandard, SurfaceUniform


class ProceduralMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        p = si.world_pos
        r = torch.exp(torch.sin(p.x) + torch.cos(p.y) - 2)
        g = torch.exp(torch.sin(p.y) + torch.cos(p.z) - 2)
        b = torch.exp(torch.sin(p.z) + torch.cos(p.x) - 2)
        v = torch.where(p.z > 0.5, 0.5 * torch.exp(-p.x * p.x * p.y * p.y * 1000), 0)
        albedo = float3(r, g, b) + v
        return SurfaceOutputStandard(albedo=albedo)
```

We can use the quick-start example for visualization:

```python
import trimesh.creation
from diffrp.scene import Scene, MeshObject
from diffrp.utils import gpu_f32, gpu_i32, to_pil
from diffrp.rendering.camera import PerspectiveCamera
from diffrp.rendering.surface_deferred import SurfaceDeferredRenderSession

mesh = trimesh.creation.icosphere(radius=0.8)
scene = Scene().add_mesh_object(MeshObject(ProceduralMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
camera = PerspectiveCamera(h=2048, w=2048)
rp = SurfaceDeferredRenderSession(scene, camera)
to_pil(rp.albedo_srgb()).save("procedural-albedo.png")
```

Note that we convert the values from linear to sRGB for better visuals.

```{figure} assets/procedural-albedo.png
:scale: 50 %
:alt: Procedural albedo material.
```

## Neural Networks

As the input and outputs are expect to be a batch of features, and they are already in native PyTorch tensor interfaces, it is quite easy to integrate a neural network in DiffRP. It is just like calling the other functions.

In this `NeuralMaterial`, we create a neural network and evaluate it for albedo output.

```python
class NeuralMaterial(SurfaceMaterial):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 3),
            torch.nn.Sigmoid()
        ).cuda()

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(albedo=self.net(si.world_normal))
```

We use the same testing code:

```python
mesh = trimesh.creation.icosphere(radius=0.8)
material = NeuralMaterial()
scene = Scene().add_mesh_object(MeshObject(material, gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
camera = PerspectiveCamera(h=512, w=512)
rp = SurfaceDeferredRenderSession(scene, camera)
to_pil(rp.albedo_srgb()).save("neural-albedo.png")
```

```{figure} assets/neural-albedo.png
:scale: 50 %
:alt: Neural albedo material.
```

You can also back-propagate the result -- it just works.

```python
pred = rp.albedo_srgb()
torch.nn.functional.mse_loss(pred, torch.rand_like(pred)).backward()
print(material.net[0].weight.grad)
```

Output:

```
tensor([[ 4.9758e-05,  4.8679e-05,  3.0747e-04],
        [-1.9968e-05,  1.7120e-06,  3.6997e-04],
        [-1.5356e-05,  1.8661e-05,  1.6705e-04],
        [-2.2664e-05, -1.2767e-05, -7.8220e-05],
        [ 8.0282e-06,  1.1283e-05, -1.3932e-04],
        [-3.6454e-06,  4.1644e-05,  2.5145e-04],
        [-2.8950e-06, -8.8152e-06, -3.0734e-04],
        [ 2.8373e-05,  2.3682e-05,  6.7197e-04],
        [-2.3877e-05,  5.6425e-05,  5.7738e-04],
        [ 4.8350e-05, -9.0493e-05, -1.0172e-03],
        [-1.4312e-05, -4.0098e-05,  2.5461e-04],
        [ 7.0399e-05,  2.5832e-05,  4.8466e-04],
        [-3.7648e-05, -2.5028e-05, -1.4115e-04],
        [ 7.4847e-05,  1.7914e-05,  5.5394e-04],
        [ 3.1179e-06, -4.5348e-06,  1.2603e-05],
        [ 7.7181e-05,  3.2846e-05, -7.4523e-04]], device='cuda:0')
```

In most cases, neural networks are agnostic to color spaces, so it will make little difference if we assume it is in sRGB or linear space (by converting or not converting the color space at output).

You may also apply losses on raw outputs from this neural network:

```python
pred = rp.albedo()
torch.nn.functional.mse_loss(pred, torch.rand_like(pred)).backward()
```
