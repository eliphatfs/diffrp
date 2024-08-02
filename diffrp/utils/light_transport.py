import math
import torch
import torch.nn.functional as F
from typing import Optional
from ..utils.shader_ops import normalized, float2, float3, cross, dot, sample2d, saturate, to_bchw, to_hwc


@torch.jit.script
def radical_inverse_van_der_corput(bits: torch.LongTensor):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.float() * 2.3283064365386963e-10


@torch.jit.script
def hammersley(n: int, deterministic: bool = False, device: Optional[torch.device] = None):
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


@torch.jit.script
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
    up = torch.where(
        n[..., 1:2] < 0.999,
        torch.tensor([0, 1, 0], dtype=n.dtype, device=n.device),
        torch.tensor([1, 0, 0], dtype=n.dtype, device=n.device)
    )
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
    for i in range(1, num_levels):
        H = H // 2
        W = W // 2
        roughness = i / (num_levels - 1)
        phi = torch.linspace(math.pi / 2 - (math.pi / 4 / H), -math.pi / 2 + (math.pi / 4 / H), H, dtype=env.dtype, device=env.device)[..., None]
        theta = torch.arange(W, dtype=env.dtype, device=env.device) * (math.tau / W)
        z = torch.cos(theta) * torch.cos(phi)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(phi)
        r = float3(x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1))  # h, w, 3

        n = v = r
        x, y = hammersley(num_samples, deterministic, env.device)  # n
        h = importance_sample_ggx(x, y, n, roughness)  # n, h, w, 3

        L = normalized(2.0 * dot(v, h) * h - v)  # broadcast, n, h, w, 3
        del h
        n_dot_L = saturate(dot(n, L))  # n, h, w, 1

        u = torch.atan2(L.x, L.z) * (0.5 / math.pi)
        v = torch.asin(torch.clamp(L.y, -1, 1)) / math.pi + 0.5
        del L
        u = u % 1

        uv = float2(u, v)
        del u, v
        values = sample2d(env, uv, "reflection")  # n, h, w, c
        del uv
        weights = n_dot_L.sum(0)

        values = torch.einsum('nhwc,nhw->hwc', values, n_dot_L.squeeze(-1))
        levels.append(values / (weights + 1e-4))
    return levels
