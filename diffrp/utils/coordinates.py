import math
import torch
from .shader_ops import float4, fsa


def near_plane_ndc_grid(H, W, dtype, device):
    y = fsa(2 / H, (torch.arange(H, dtype=dtype, device=device))[..., None, None], 1 / H - 1)
    x = fsa(2 / W, (torch.arange(W, dtype=dtype, device=device))[..., None], 1 / W - 1)
    return float4(x, y, -1, 1)


def latlong_grid(H, W, dtype, device):
    phi = torch.linspace(math.pi / 2, -math.pi / 2, H, dtype=dtype, device=device)[..., None, None]
    theta = (torch.arange(W, dtype=dtype, device=device))[..., None] * (math.tau / (W - 1))
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
    theta = torch.atan2(L.x, L.z)
    phi = torch.asin(torch.clamp(L.y, -1, 1))
    return theta, phi


def unit_direction_to_latlong_uv(L: torch.Tensor):
    u = (torch.atan2(L.x, L.z) * (0.5 / math.pi)) % 1
    v = fsa(1 / math.pi, torch.asin(torch.clamp(L.y, -1, 1)), 0.5)
    return u, v
