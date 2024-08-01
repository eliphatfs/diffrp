import math
import torch
import torch_redstone as rst
import matplotlib.pyplot as plotlib
from diffrp.utils.shader_ops import normalized


def radical_inverse_van_der_corput(bits: torch.Tensor):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.float() * 2.3283064365386963e-10


def hammersley(n: int, device='cuda'):
    i = torch.arange(n, dtype=torch.int64, device=device)
    x = i.float() * (1 / n)
    y = radical_inverse_van_der_corput(i)
    return x, y


def importance_sample_ggx(x: torch.Tensor, y: torch.Tensor, n: torch.Tensor, roughness: torch.Tensor):
    a = roughness * roughness
    phi = math.tau * x
    cos_theta = torch.sqrt((1 - y) / (1 + (a * a - 1) * y))
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
    h = torch.stack([
        torch.cos(phi) * sin_theta,
        torch.sin(phi) * sin_theta,
        cos_theta
    ])
    up = torch.where(n.z < 0.999, n.new_tensor([0, 1, 0]), n.new_tensor([1, 0, 0]))
    tangent = normalized(torch.cross(up, n, dim=-1))
    bitangent = torch.cross(n, tangent, dim=-1)
    sample_vec = tangent * h.x + bitangent * h.y + n * h.z
    return normalized(sample_vec)
