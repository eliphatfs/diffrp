import math
import torch


def latlong_grid(H, W, dtype, device):
    phi = torch.linspace(math.pi / 2, -math.pi / 2, H, dtype=dtype, device=device)[..., None, None]
    theta = (torch.arange(W, dtype=dtype, device=device))[..., None] * (math.tau / (W - 1))
    return phi, theta


def angles_to_unit_direction(theta: torch.Tensor, phi: torch.Tensor):
    return _angles_to_unit_direction_impl(theta, phi)


@torch.jit.script
def _angles_to_unit_direction_impl(theta: torch.Tensor, phi: torch.Tensor):
    z = torch.cos(theta) * torch.cos(phi)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(phi)
    return x, y, z


def unit_direction_to_angles(L: torch.Tensor):
    theta = torch.atan2(L.x, L.z)
    phi = torch.asin(torch.clamp(L.y, -1, 1))
    return theta, phi


def unit_direction_to_latlong_uv(L: torch.Tensor):
    u = (torch.atan2(L.x, L.z) * (0.5 / math.pi)) % 1
    v = torch.asin(torch.clamp(L.y, -1, 1)) / math.pi + 0.5
    return u, v
