# Comparison Between Pipelines

Currently, DiffRP supports two rendering pipelines:
deferred lighting rasterization pipeline ({py:class}`diffrp.rendering.surface_deferred.SurfaceDeferredRenderSession`)
and path tracing pipeline ({py:class}`diffrp.rendering.path_tracing.PathTracingSession`).

The deferred lighting rasterization pipeline ({py:class}`diffrp.rendering.surface_deferred.SurfaceDeferredRenderSession`) supports PBR with image-based lighting,
and can render various useful attributes like depth, world/camera-space normal, albedo, metallic/roughness attributes and so on.
It supports analytical antialiasing and edge gradient methods as the visibility gradient.
It requires ``nvdiffrast`` to be installed, which requires a working CUDA toolkit.

The path tracing pipeline ({py:class}`diffrp.rendering.path_tracing.PathTracingSession`) is mainly targetting unbiased PBR.
While the PBR rendering will produce albedo, world normal and first-hit positions as a by-product,
rendering other attributes are not targeted by default.
It is based on monte-carlo sampling of paths, so the PBR result would contain noise.
You can use the denoiser, which is a differentiable version of a denoiser in OIDN, to denoise the result.
It does not support common visibility gradient handling methods yet.
It requires ``torchoptix`` to be installed for the best performance using RTX hardware, but can also run in pure PyTorch.
