import math
import torch
from .shader_ops import flipper_2d


def near_plane_ndc_grid(H: int, W: int, dtype: torch.dtype, device: torch.device):
    y = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, dtype=dtype, device=device).view(H, 1, 1).expand(H, W, 1)
    x = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, dtype=dtype, device=device).view(1, W, 1).expand(H, W, 1)
    near = (-flipper_2d()).expand(H, W, 2)
    return torch.cat([x, y, near], dim=-1)


def latlong_grid(H, W, dtype, device):
    phi = torch.linspace(math.pi / 2, -math.pi / 2, H, dtype=dtype, device=device).view(-1, 1, 1)
    theta = torch.arange(W, dtype=dtype, device=device).view(-1, 1) * (math.tau / (W - 1))
    return phi, theta


def angles_to_unit_direction(theta: torch.Tensor, phi: torch.Tensor):
    """
    Convert angle tensors (theta, phi) into unit direction vectors (x, y, z).

    The method is broadcastable.

    Inputs angles are in radians, unlike ``from_orbit``.

    See also:
        :py:obj:`diffrp.rendering.camera.PerspectiveCamera.from_orbit`.
    """
    return _angles_to_unit_direction_impl(theta, phi)


@torch.jit.script
def _angles_to_unit_direction_impl(theta: torch.Tensor, phi: torch.Tensor):
    z = torch.cos(theta) * torch.cos(phi)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(phi)
    return x, y, z


def unit_direction_to_angles(L: torch.Tensor):
    """
    Convert a unit direction vector (...batch_dims, 3) into two angle tensors (theta, phi).
    Theta and phi refer to the azimuth and elevation, respectively.

    Theta and phi will have one fewer dimension, shape (...batch_dims).
    These angles are in radians, unlike ``from_orbit``.

    See also:
        :py:obj:`diffrp.rendering.camera.PerspectiveCamera.from_orbit`.
    """
    z = L.z
    if L.requires_grad:
        z = torch.where(z == 0, 1e-8, z)
    theta = torch.atan2(L.x, z)
    phi = torch.asin(torch.clamp(L.y, -0.999999, 0.999999))
    return theta, phi


@torch.jit.script
def _unit_direction_to_latlong_uv_impl(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    u = (torch.atan2(x, z) * (0.5 / math.pi)) % 1
    v = (1 / math.pi) * torch.asin(torch.clamp(y, -0.999999, 0.999999)) + 0.5
    return u, v


def unit_direction_to_latlong_uv(L: torch.Tensor):
    z = L.z
    if L.requires_grad:
        z = torch.where(z == 0, 1e-8, z)
    return _unit_direction_to_latlong_uv_impl(L.x, L.y, z)
