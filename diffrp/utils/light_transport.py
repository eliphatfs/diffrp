import math
import torch
import torch.nn.functional as F
from typing import Optional
from .coordinates import angles_to_unit_direction, unit_direction_to_latlong_uv
from .shader_ops import normalized, float2, float3, cross, dot, sample2d, saturate, to_bchw, to_hwc


@torch.jit.script
def radical_inverse_van_der_corput(bits: torch.LongTensor):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.float() * 2.3283064365386963e-10


def hammersley(n: int, deterministic: bool = False, device: Optional[torch.device] = None):
    return _hammersley_impl(n, deterministic, device)


@torch.jit.script
def _hammersley_impl(n: int, deterministic: bool, device: Optional[torch.device]):
    i = torch.arange(n, dtype=torch.int64, device=device)
    x = i.float() * (1 / n)
    y = radical_inverse_van_der_corput(i)
    if not deterministic:
        x = (x + torch.rand(1, dtype=x.dtype, device=x.device)) % 1.0
        y = (y + torch.rand(1, dtype=x.dtype, device=x.device)) % 1.0
    return x, y


def normal_distribution_function_ggx(n_dot_h: torch.Tensor, roughness: float):
    a = roughness * roughness
    f = n_dot_h * n_dot_h * (a * a - 1) + 1
    return (a * a / math.pi + 1e-7) / (f * f + 1e-7)


def importance_sample_ggx(x: torch.Tensor, y: torch.Tensor, n: torch.Tensor, roughness: float):
    """
    GGX Importance Sampling.

    Note:
        Due to different roughness values usually require different resolutions of `n`,
        we do not support batched roughness values in this function.

    Args:
        x (torch.Tensor): sample sequence element 1, shape (n) in [0, 1)
        y (torch.Tensor): sample sequence element 2, shape (n) in [0, 1)
        n (torch.Tensor): (batched) normal vectors, shape (..., 3)
        roughness (float): the roughness level
    
    Returns:
        torch.Tensor: sampled ray directions, shape (n, ..., 3)
    """
    return _importance_sample_ggx_impl(x, y, n, roughness)


@torch.jit.script
def _importance_sample_ggx_impl(x: torch.Tensor, y: torch.Tensor, n: torch.Tensor, roughness: float):
    a = roughness * roughness
    phi = math.tau * x
    if roughness < 1e-4:
        cos_theta = torch.ones_like(y)
    else:
        cos_theta = torch.sqrt((1 - y) / (1 + (a * a - 1) * y))
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
    hx = torch.cos(phi) * sin_theta
    hy = torch.sin(phi) * sin_theta
    hz = cos_theta
    up = torch.eye(3, dtype=n.dtype, device=n.device)[torch.where(
        n[..., 1] < 0.999, 1, 0
    )]
    # cannot use .y and .new_tensor shorthands for jit scripted function
    tangent = normalized(cross(up, n))
    bitangent = cross(n, tangent)
    # n: ..., 3
    # h*: n
    h_broadcast = [-1] + [1] * n.ndim
    hx, hy, hz = [m.reshape(h_broadcast) for m in [hx, hy, hz]]
    sample_vec = tangent * hx + bitangent * hy + n * hz
    return normalized(sample_vec)


def prefilter_env_map(
    env: torch.Tensor,
    base_resolution: int = 128, num_levels: int = 5,
    num_samples: int = 1024, deterministic: bool = False
):
    H = base_resolution
    W = H * 2
    env = to_hwc(F.interpolate(to_bchw(env), [H, W], mode='area'))
    levels = [env]
    rx, ry = hammersley(num_samples, deterministic, env.device)  # n
    for i in range(1, num_levels):
        H = H // 2
        W = W // 2
        roughness = i / (num_levels - 1)
        phi = torch.linspace(math.pi / 2 - (math.pi / 4 / H), -math.pi / 2 + (math.pi / 4 / H), H, dtype=env.dtype, device=env.device)[..., None, None]
        theta = (torch.arange(W, dtype=env.dtype, device=env.device) + 0.5)[..., None] * (math.tau / W)
        x, y, z = angles_to_unit_direction(theta, phi)
        r = float3(x, y, z)  # h, w, 3

        n = v = r
        h = importance_sample_ggx(rx, ry, n, roughness)  # n, h, w, 3

        L = normalized(2.0 * dot(v, h) * h - v)  # broadcast, n, h, w, 3
        del h
        n_dot_L = torch.relu_(dot(n, L))  # n, h, w, 1

        u, v = unit_direction_to_latlong_uv(L)
        del L

        uv = float2(u, v)
        del u, v
        values = sample2d(env, uv, "reflection")  # n, h, w, c
        del uv
        weights = n_dot_L.sum(0)

        values = torch.einsum('nhwc,nhw->hwc', values, n_dot_L.squeeze(-1))
        levels.append(values / (weights + 1e-4))
    return levels
