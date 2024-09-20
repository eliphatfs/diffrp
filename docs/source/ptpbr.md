# Path-Traced Physically Based Rendering

DiffRP implements a path tracing pipeline for high-quality physically-based rendering.

We first load the [ABeautifulGame](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ABeautifulGame) scene and add a environment light, as in the basic GLTF tutorial:

```python
from diffrp.resources import hdris
from diffrp.scene import ImageEnvironmentLight

scene = load_gltf_scene("beautygame.glb", compute_tangents=True).static_batching()
scene.add_light(ImageEnvironmentLight(
    intensity=1.0, color=torch.ones(3, device='cuda'),
    image=hdris.newport_loft().cuda(), render_skybox=True
))
```

Create a camera to look at a beautiful glazing angle:

```python
from diffrp.rendering.camera import PerspectiveCamera
camera = PerspectiveCamera.from_orbit(
    h=720, w=1280,  # resolution
    radius=0.9, azim=50, elev=10,  # orbit camera position, azim/elev in degrees
    origin=[0, 0, 0],  # orbit camera focus point
    fov=28.12, near=0.005, far=100.0  # intrinsics
)
```

Create the rendering session and perform rendering:

```python
from diffrp.rendering.path_tracing import PathTracingSession, PathTracingSessionOptions
rp = PathTracingSession(scene, camera, PathTracingSessionOptions(ray_spp=8))
pbr, alpha, extras = rp.pbr()
```

Perform tone-mapping and view the image:

```python
from diffrp.utils import to_pil, agx_base_contrast
pbr_mapped = agx_base_contrast(pbr)
to_pil(pbr_mapped).save("game.png")
```

```{figure} assets/game-8spp-noisy.jpg
:scale: 45 %
:alt: Game noisy PBR render.
```

Due to the monte-carlo sampling there is noise in the image.
You can use a higher `ray_spp` to get a less noisy image, and you can also use a denoiser.

DiffRP provides a differentiable version of the HDR-ALB-NML OIDN denoiser. You can even train the denoiser if you would like! 

```python
from diffrp.rendering.denoiser import get_denoiser, run_denoiser
from diffrp.utils import linear_to_srgb
denoiser = get_denoiser()
pbr_denoised = agx_base_contrast(run_denoiser(denoiser, pbr, linear_to_srgb(extras['albedo']), extras['world_normal']))
to_pil(pbr_denoised).save("gamed.png")
```

```{figure} assets/game-1024spp-denoised.jpg
:scale: 45 %
:alt: Game denoised PBR render.
```

Transparent film is not well supported for now, `render_skybox` will not affect the background region in the radiance output.
You can still combine with the `alpha` output to obtain a transparent film image, but you may see some background bleeding into the edges.

```python
from diffrp.utils import float4
to_pil(float4(pbr_denoised, alpha)).save("gameda.png")
```

```{figure} assets/game-8spp-denoised-alpha.jpg
:scale: 45 %
:alt: Game denoised PBR render with transparent film.
```

Complete example:

```python
import torch
from diffrp.resources import hdris
from diffrp.scene import ImageEnvironmentLight
from diffrp.rendering.camera import PerspectiveCamera
from diffrp.loaders.gltf_loader import load_gltf_scene
from diffrp.utils import to_pil, agx_base_contrast, linear_to_srgb, float4
from diffrp.rendering.path_tracing import PathTracingSession, PathTracingSessionOptions
from diffrp.rendering.denoiser import get_denoiser, run_denoiser

torch.set_grad_enabled(False)
scene = load_gltf_scene("beautygame.glb", True).static_batching()
scene.add_light(ImageEnvironmentLight(
    intensity=1.0, color=torch.ones(3, device='cuda'),
    image=hdris.newport_loft().cuda(), render_skybox=True
))
camera = PerspectiveCamera.from_orbit(
    h=720, w=1280,  # resolution
    radius=0.9, azim=50, elev=10,  # orbit camera position, azim/elev in degrees
    origin=[0, 0, 0],  # orbit camera focus point
    fov=28.12, near=0.005, far=100.0  # intrinsics
)
rp = PathTracingSession(scene, camera, PathTracingSessionOptions(ray_spp=8))
pbr, alpha, extras = rp.pbr()
denoiser = get_denoiser()
pbr_mapped = agx_base_contrast(pbr)
pbr_denoised = agx_base_contrast(run_denoiser(denoiser, pbr, linear_to_srgb(extras['albedo']), extras['world_normal']))
to_pil(pbr_mapped).save("game.png")
to_pil(pbr_denoised).save("gamed.png")
to_pil(float4(pbr_denoised, alpha)).save("gameda.png")
```
