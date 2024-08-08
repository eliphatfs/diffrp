import math
import torch
import torch.nn.functional as F
from typing import Optional
from .cache import singleton_cached
from .coordinates import angles_to_unit_direction, unit_direction_to_latlong_uv, latlong_grid
from .shader_ops import normalized, float2, float3, cross, dot, sample2d, to_bchw, to_hwc, saturate, sample3d, fma


@torch.jit.script
def radical_inverse_van_der_corput(bits: torch.Tensor):
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
    return (a * a / math.pi) / (f * f)


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
    return sample_vec


def prefilter_env_map(
    env: torch.Tensor,
    base_resolution: int = 256, num_levels: int = 5,
    num_samples: int = 512, deterministic: bool = False
):
    H = base_resolution
    W = H * 2
    env = to_hwc(F.interpolate(to_bchw(env), [H, W], mode='area'))
    pre_mips = [env]
    while H > 8:
        H = H // 2
        W = W // 2
        pre_mips.append(to_hwc(F.interpolate(to_bchw(pre_mips[-1]), [H, W], mode='area')))
    H = base_resolution
    W = H * 2
    pre_mips = torch.stack([to_hwc(F.interpolate(to_bchw(mip), [H, W], mode='bilinear')) for mip in pre_mips])
    levels = [env]
    rx, ry = hammersley(num_samples, deterministic, env.device)  # n
    for i in range(1, num_levels):
        H = H // 2
        W = W // 2
        roughness = i / (num_levels - 1)
        phi, theta = latlong_grid(H, W, env.dtype, env.device)
        x, y, z = angles_to_unit_direction(theta, phi)
        r = float3(x, y, z)  # h, w, 3

        n = v = r
        h = importance_sample_ggx(rx, ry, n, roughness)  # n, h, w, 3
        ggx_d = normal_distribution_function_ggx(dot(n, h), roughness)  # n, h, w, 1

        L = normalized(2.0 * dot(v, h) * h - v)  # broadcast, n, h, w, 3
        del h
        n_dot_L = torch.relu_(dot(n, L))  # n, h, w, 1

        u, v = unit_direction_to_latlong_uv(L)
        satexel_inv = (2.0 * base_resolution * base_resolution) / math.pi
        sasample = (satexel_inv / num_samples) / (torch.clamp_min(1 - L.y ** 2, math.cos(math.pi / 2 - math.pi / 2 / H) ** 2) * (ggx_d + 0.0004))
        del ggx_d, L

        mip_level = saturate(0.5 / (len(pre_mips) - 1) * torch.log2(sasample))
        del sasample

        uv = float3(u, v, mip_level)
        del u, v, mip_level
        values = sample3d(pre_mips, uv, "reflection")  # n, h, w, c
        del uv
        weights = n_dot_L.sum(0)

        values = torch.einsum('nhwc,nhw->hwc', values, n_dot_L.squeeze(-1))
        values = values / (weights + 1e-4) 
        values = to_hwc(F.interpolate(to_bchw(values), [base_resolution, base_resolution * 2], mode='bilinear', align_corners=True))
        levels.append(values)
    levels.reverse()
    return torch.stack(levels)


def irradiance_integral_env_map(
    env: torch.Tensor,
    premip_resolution: int = 64,
    base_resolution: int = 16
):
    HP, WP = premip_resolution, premip_resolution * 2
    env = to_hwc(F.interpolate(to_bchw(env), [HP, WP], mode='area'))
    H, W = base_resolution, base_resolution * 2
    phi, theta = latlong_grid(H, W, env.dtype, env.device)
    x, y, z = angles_to_unit_direction(theta, phi)
    n = float3(x, y, z)[..., None, None, :]  # h, w, 3 -> h, w, 1, 1, 3
    up = torch.eye(3, dtype=env.dtype, device=env.device)[1].expand_as(n)
    right = normalized(cross(up, n))
    up = normalized(cross(n, right)) # h, w, 1, 1, 3
    sample_phi, sample_theta = latlong_grid(HP * 2, WP * 2, env.dtype, env.device)
    sample_phi = sample_phi[:len(sample_phi) // 2]
    x, y, z = angles_to_unit_direction(sample_theta, sample_phi)  # tangent space, hp, wp, 1
    uv = float2(*unit_direction_to_latlong_uv(x * right + z * up + y * n))  # h, w, hp, wp, 2
    values = sample2d(env, uv) * torch.cos(sample_phi) * torch.sin(sample_phi)  # h, w, hp, wp, 3
    # sample_phi: hp, 1, 1
    values = values.mean([-2, -3])
    return values * math.pi


def geometry_schlick_ggx(n_dot_v: torch.Tensor, roughness: torch.Tensor):
    a = roughness
    k = (a * a) / 2.0
    nom = n_dot_v
    denom = n_dot_v * (1.0 - k) + k
    return nom / denom


def geometry_smith(n: torch.Tensor, v: torch.Tensor, L: torch.Tensor, roughness: torch.Tensor):
    n_dot_v = torch.relu_(dot(n, v))
    n_dot_L = torch.relu_(dot(n, L))
    ggx2 = geometry_schlick_ggx(n_dot_v, roughness)
    ggx1 = geometry_schlick_ggx(n_dot_L, roughness)
    return ggx1 * ggx2


def fresnel_schlick_smoothness(cos_theta: torch.Tensor, f0: torch.Tensor, smoothness: torch.Tensor):
    return fma(torch.relu_(smoothness - f0), (1.0 - cos_theta) ** 5.0, f0)


@singleton_cached
def pre_integral_env_brdf():
    device = 'cuda'
    s = 256
    n_dot_v = torch.linspace(0 + 0.5 / s, 1 - 0.5 / s, s, device=device, dtype=torch.float32)[..., None]

    v = float3(torch.sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v)
    n = n_dot_v.new_tensor([0, 0, 1]).expand_as(v)

    rx, ry = hammersley(1024, True, device)  # n
    results = []
    for i in range(256):
        roughness = (i + 0.5) / 256
        h = importance_sample_ggx(rx, ry, n, roughness)  # n, s, 3
        L = normalized(2 * dot(v, h) * h - v)
        # n_dot_L = torch.relu(L.z)
        n_dot_h = torch.relu(h.z)
        v_dot_h = torch.relu(dot(v, h))
        g = geometry_smith(n, v, L, roughness)
        g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v)
        fc = (1 - v_dot_h) ** 5
        results.append((g_vis * float2(1 - fc, fc)).mean(0))
    return torch.stack(results)
