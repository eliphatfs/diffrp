# Common Concepts

## Scene

A *scene* includes objects and lights.
The data structure can be found in {py:class}`diffrp.scene.scene.Scene`.

Currently, only `MeshObject`s are supported for objects, and only `ImageEnvironmentLight`s are supported for lights.

**Mesh Object**

A `MeshObject` is a conventional triangle mesh formed by vertices and triangles, associated with a *material* and optionally vertex attributes including vertex normals, tangents, colors, UVs and other custom attributes.

## Material

A *material* defines the appearance of an object. In DiffRP, materials are designed to be compatible across different render pipelines, though there is only one pipeline implemented currently. As only mesh objects are considered, there is only the `SurfaceMaterial` base that defines appearances of surfaces now, or more theoretically, how each point on the surfaces interact with light.

Materials can be thought as functions defined on a point on the surface. The point may be associated with a triangle on the mesh, a world coordinate, and many extensive attributes like vertex normals, UVs and so on. How a point on the surface interacts with light is local -- it should not be affected by other points on the surface, or lights in the scene. If you need interaction with the scene outside the `SurfaceInput` and `SurfaceUniform` objects you have in the material function, it would in most cases be better that you rethink about it and do it by extending the render pipelines.

## Color Spaces

Physically correct light interactions should be computed in the linear space since light is linear -- light is made of photons, and photons behave the same as each other. Linear means linear in number of photons, which is linear to the energy of light.

Thus, color data is assumed linear in DiffRP, including those loaded by {py:mod}`diffrp.loaders.gltf_loader`.

However, this is not how our eyes work. The perception of human vision is in the sRGB space, which is a different space than linear. Most LDR images (e.g. JPEG/PNG) are thus stored in the sRGB space on the computers. The mapping is commonly wrongly referred to as *gamma correction* but it should be a slightly different mapping. You can find it in {py:mod}`diffrp.utils.color`.

You may need to convert input images (like environment images or texture maps in JPEG/PNG) from sRGB to Linear before doing PBR, and convert the output images from Linear to sRGB before you view them. EXR/HDR images are usually already in linear space.

If you are not doing PBR but only working on processing various attributes or fitting baked colors without considering interations with lights, color spaces are not that much -- theoretically, the correct interpolation requires linear spaces but usually the error is also close to invisible in sRGB. You just make sure your are in the same space, or in any space you expect.

## Tone Mapping

There is no cap on the energy of light in real world. Thus, you usually get HDR images with maximum levels larger than 1 during PBR. To output it into a regular PNG image, you not only need to convert the color space from linear to sRGB, but also deal with these beyond-the-limit values.

A naive way is to clip the values larger than 1.0. While this can work to some degree, we usually observe unnatural saturated and overexposed regions with this. By the way, this is usually the 'None' tone-mapping in rendering engines if you happen to see these options in unity or blender.

The correct way is to apply a dedicated tone mapper. DiffRP provides a differentiable implementation of the state-of-the-art AgX tone mapper in {py:mod}`diffrp.utils.tone_mapping`. This usually results in much more realistic images than naive clipping.

## Visibility Gradient

If you move a triangle by a small distance, colors on its edge can be affected, thus this effect should contribute to the gradient of the render. However, the edge of a triangle is a discontinuous region in the function, leading to difficulties in automatic differentiation.

Visibility gradient is thus referring to techniques to take account of this contribution, and is special to differentiable rendering.

The surface deferred render pipeline in DiffRP implements the Edge Gradient and the anti-aliasing approximation from nvdiffrast as visibility gradient generators.
