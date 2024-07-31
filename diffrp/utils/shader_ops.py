import numpy
import torch
import operator
import functools
import itertools
import torch_redstone as rst
import torch.nn.functional as F


def gpu_f32(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dtype=torch.float32, device='cuda').contiguous()
    return torch.tensor(inputs, dtype=torch.float32, device='cuda').contiguous()


def gpu_i32(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dtype=torch.int32, device='cuda').contiguous()
    return torch.tensor(inputs, dtype=torch.int32, device='cuda').contiguous()


def gpu_color(inputs):
    if inputs is None:
        return None
    if not isinstance(inputs, torch.Tensor):
        inputs = numpy.array(inputs)
    x = gpu_f32(inputs)
    if x.max() > 1.5:
        x /= 255.0
    return x


def multi_surf(*surfs):
    return lambda x: tuple(surf(x) for surf in surfs)


def zeros_like_vec(x: torch.Tensor, v):
    return x.new_zeros(*x.shape[:-1], v)


def ones_like_vec(x: torch.Tensor, v):
    return x.new_ones(*x.shape[:-1], v)


def full_like_vec(x: torch.Tensor, value, v):
    return x.new_full([*x.shape[:-1], v], value)


def homogeneous(coords: torch.Tensor):
    return torch.cat([coords, ones_like_vec(coords, 1)], dim=-1)


def homogeneous_vec(coords: torch.Tensor):
    return torch.cat([coords, zeros_like_vec(coords, 1)], dim=-1)


def transform_point(xyz, matrix):
    h = homogeneous(xyz)
    return torch.matmul(h, matrix.T)


def transform_point4x3(xyz, matrix):
    return torch.addmm(matrix[:-1, -1], xyz, matrix[:-1, :-1].T)


def transform_vector(xyz, matrix):
    h = homogeneous_vec(xyz)
    return torch.matmul(h, matrix.T)[..., :3]


def transform_vector3x3(xyz, matrix):
    return torch.matmul(xyz, matrix[:-1, :-1].T)


def sample2d(
    texture2d: torch.Tensor,
    texcoords: torch.Tensor,
    wrap: str = "border",
    mode: str = "bilinear"
):
    # bhwc -> bchw
    # texcoords: ..., 2
    sampled = F.grid_sample(
        torch.flipud(texture2d)[None].permute(0, 3, 1, 2),
        texcoords.flatten(0, -2)[None, None] * 2 - 1,
        padding_mode=wrap, mode=mode, align_corners=False
    ).squeeze(2).squeeze(0).T
    # bchw -> wc -> ..., c
    return sampled.reshape(*texcoords.shape[:-1], texture2d.shape[-1]).contiguous()


def saturate(x: torch.Tensor):
    return torch.clamp(x, 0.0, 1.0)


def split_alpha(x: torch.Tensor):
    return x[..., :-1], x[..., -1:]


def normalized(x: torch.Tensor):
    return F.normalize(x, dim=-1)


def length(x: torch.Tensor):
    return torch.norm(x, dim=-1, keepdim=True)


def floatx(*tensors):
    tensors = [x if torch.is_tensor(x) else gpu_f32(x) for x in tensors]
    tensors = rst.supercat(tensors, dim=-1)
    return tensors


def float4(*tensors):
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 4
    return tensors


def float3(*tensors):
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 3
    return tensors


def float2(*tensors):
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 2
    return tensors


def to_bchw(hwc: torch.Tensor):
    if hwc.ndim == 3:
        hwc = hwc[None]
    return hwc.permute(0, 3, 1, 2)


def to_c1hw(hwc: torch.Tensor):
    return hwc.permute(2, 0, 1)[:, None]


def from_c1hw(c1hw: torch.Tensor):
    return c1hw.squeeze(1).permute(1, 2, 0)


def to_hwc(bchw: torch.Tensor):
    if bchw.ndim == 3:
        bchw = bchw[None]
    return bchw.permute(0, 2, 3, 1).squeeze(0)


def blur2d(texture2d: torch.Tensor, size: int, iters: int):
    tex = to_c1hw(texture2d)
    ks = 2 * size + 1
    ker = tex.new_ones([1, 1, ks, ks]) / ks ** 2
    for _ in range(iters):
        tex = torch.conv2d(tex, ker, padding=size)
    return from_c1hw(tex)


def sobel(texture2d: torch.Tensor):
    tex = to_c1hw(texture2d)
    kx = tex.new_tensor([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]).reshape(1, 1, 3, 3)
    ky = tex.new_tensor([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).reshape(1, 1, 3, 3)
    fx = from_c1hw(torch.conv2d(tex, kx, padding=1))
    fy = from_c1hw(torch.conv2d(tex, ky, padding=1))
    return float4(fx, fy, (fx ** 2 + fy ** 2) ** 0.5, texture2d.x)


def rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:
    return rgb2hsv_internal(rgb.flatten(0, -2)).reshape(rgb.shape)


def hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    return hsv2rgb_internal(hsv.flatten(0, -2)).reshape(hsv.shape)


def rgb2hsv_internal(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_internal(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


@functools.lru_cache(maxsize=None)
def white_tex():
    return gpu_f32(numpy.ones([16, 16, 4], dtype=numpy.float32))


@functools.lru_cache(maxsize=None)
def black_tex():
    return gpu_f32(numpy.zeros([16, 16, 4], dtype=numpy.float32))


@functools.lru_cache(maxsize=None)
def gray_tex():
    return gpu_f32(numpy.full([16, 16, 4], 0.5, dtype=numpy.float32))


@functools.lru_cache(maxsize=None)
def empty_normal_tex():
    return gpu_f32(numpy.full([16, 16, 3], [0.5, 0.5, 1.0], dtype=numpy.float32))


def _make_attr(attr: str, idx: str):
    catlist = [idx.index(c) for c in attr]
    for s in range(1, 4):
        if all(x - y == s for x, y in zip(catlist[1:], catlist)):
            # fast, shape operation
            indexer = slice(catlist[0], catlist[-1] + 1, s)
            break
    else:
        # general but slow, has cpu-gpu synchronization and allocation
        indexer = catlist
    setattr(torch.Tensor, attr, property(operator.itemgetter((..., indexer))))


indices = ['xyzw', 'rgba']
for idx in indices:
    comb = [''.join(x) for x in itertools.product([''] + list(idx), repeat=len(idx))]
    comb = [x for x in comb if x]
    for attr in comb:
        _make_attr(attr, idx)
