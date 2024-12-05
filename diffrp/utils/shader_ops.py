"""
GLSL Attribute Access
=======================

In addition to the utility methods,
the module also contain a syntactic sugar to access channels of the last dimension
in tensors conveniently, in a GLSL-like manner.

This would be particularly useful for graphics applications.

``tnsr.xyzw`` can access the [0, 1, 2, 3] elements of the last channel, respectively.
The same goes with ``tnsr.rgba``.

You can combine any of them according to your need like ``a.xx``, ``b.wxy``, ``c.rgb``, ``d.a``, etc..

The singleton dimensions are not removed. If ``a`` has shape (N, 3), ``a.x`` will have shape (N, 1).
"""
import numpy
import torch
import operator
import itertools
from typing import Union
import torch_redstone as rst
import torch.nn.functional as F
from .cache import singleton_cached


def gpu_f32(inputs):
    """
    Cast any input (float, list, numpy array, tensor) into
    ``torch.Tensor`` with dtype float32 on the current CUDA device.
    ``CUDA_VISIBLE_DEVICES`` and ``torch.cuda.set_device`` can be used to specify the current device.
    """
    if isinstance(inputs, (float, int)):
        return torch.full([], inputs, dtype=torch.float32, device='cuda')
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dtype=torch.float32, device='cuda')
    return torch.tensor(inputs, dtype=torch.float32, device='cuda')


def gpu_i32(inputs):
    """
    Cast any input (float, list, numpy array, tensor)
    into ``torch.Tensor`` with dtype int32 on the current CUDA device.
    ``CUDA_VISIBLE_DEVICES`` and ``torch.cuda.set_device`` can be used to specify the current device.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dtype=torch.int32, device='cuda')
    return torch.tensor(inputs, dtype=torch.int32, device='cuda')


def gpu_color(inputs):
    """
    Cast any input (float, list, numpy array, tensor) into
    ``torch.Tensor`` with dtype float32 on the current CUDA device.
    If any of the inputs have a maximum value greater than 1.5, we assume the input to be uint8 coded
    and divide the colors by 255.
    ``CUDA_VISIBLE_DEVICES`` and ``torch.cuda.set_device`` can be used to specify the current device.
    """
    if inputs is None:
        return None
    if not isinstance(inputs, torch.Tensor):
        inputs = numpy.array(inputs)
    x = gpu_f32(inputs)
    if x.max() > 1.5:
        x /= 255.0
    return x


def fsa(a: float, x: torch.Tensor, b: Union[torch.Tensor, float]):
    """
    a * x + b (scale-add), but fused/faster
    """
    return torch.add(b, x, alpha=a)


def fma(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, a: float = 1.0):
    """
    a * x * y + b (multiply-add), but fused/faster.
    """
    return torch.addcmul(b, x, y, value=a)


def multi_surf(*surfs):
    return lambda x: tuple(surf(x) for surf in surfs)


def zeros_like_vec(x: torch.Tensor, v):
    """
    Zeros like the given tensor, but change the last shape dimension to ``v``.
    """
    return x.new_zeros(*x.shape[:-1], v)


def ones_like_vec(x: torch.Tensor, v):
    """
    Ones like the given tensor, but change the last shape dimension to ``v``.
    """
    return x.new_ones(*x.shape[:-1], v)


def full_like_vec(x: torch.Tensor, value, v):
    """
    Fill in ``value`` like the given tensor, but change the last shape dimension to ``v``.
    """
    return x.new_full([*x.shape[:-1], v], value)


def homogeneous(coords: torch.Tensor):
    """
    Concatenate a channel of ones to the given ``coords`` to obtain its homogeneous-coordinate position.
    """
    return F.pad(coords, (0, 1), value=1)


def homogeneous_vec(coords: torch.Tensor):
    """
    Concatenate a channel of zeros to the given ``coords`` to obtain its homogeneous-coordinate direction.
    """
    return F.pad(coords, (0, 1), value=0)


def transform_point(xyz, matrix):
    """
    Transform points ``xyz`` with the homogeneous transform ``matrix``.

    The last dimension of ``xyz`` needs to be ``3``.
    The result will have the same shape as ``xyz`` but the last dimension changed to ``4``.

    ``matrix`` is a tensor of shape (4, 4) and the transform is applied
    as a left-multiplication to ``xyz``.

    The method is fully broadcastable to any dimensions.
    """
    h = homogeneous(xyz)
    return torch.matmul(h, matrix.T)


def transform_point4x3(xyz, matrix):
    """
    Transform points ``xyz`` with the homogeneous transform ``matrix``.
    The matrix needs to be affine.
    The last dimension of ``xyz`` needs to be ``3``.
    
    This is in general faster than doing a ``transform_point`` and then perform a homogeneous division.

    The result will have the same shape as ``xyz``, unlike ``transform_point`` which
    keeps the homogeneous coordinate form.

    ``xyz`` should be of shape (B, 3).
    """
    return torch.addmm(matrix[:-1, -1], xyz, matrix[:-1, :-1].T)


def transform_vector(xyz, matrix):
    """
    Transform a vector (direction) ``xyz`` by the homogeneous transform ``matrix``.
    The last dimension of ``xyz`` needs to be ``3``.

    The result will have the same shape as ``xyz``.

    ``matrix`` is a tensor of shape (4, 4) and the transform is applied
    as a left-multiplication to ``xyz``.

    The method is fully broadcastable to any dimensions.
    """
    h = homogeneous_vec(xyz)
    return torch.matmul(h, matrix.T)[..., :3]


def transform_vector3x3(xyz, matrix):
    """
    Transform a vector (direction) ``xyz`` by the homogeneous transform ``matrix``.
    The last dimension of ``xyz`` needs to be ``3``.

    The matrix needs to be affine.

    The result will have the same shape as ``xyz``.

    ``matrix`` is a tensor of shape (4, 4) and the transform is applied
    as a left-multiplication to ``xyz``.

    The method is fully broadcastable to any dimensions.
    """
    return torch.matmul(xyz, matrix[:-1, :-1].T)


@singleton_cached
def flipper_2d():
    return gpu_f32([1, -1])


@singleton_cached
def flipper_3d():
    return gpu_f32([1, -1, 1])


@torch.jit.script
def _sample2d_internal(
    texture2d: torch.Tensor,
    texcoords: torch.Tensor,
    flipper: torch.Tensor,
    wrap: str = "border",
    mode: str = "bilinear"
):
    texcoords = texcoords.reshape(1, 1, -1, texcoords.shape[-1])
    align_corners = False
    if wrap == "latlong":
        align_corners = True
        wrap = "reflection"
    if wrap == "cyclic":
        texture2d = texture2d.tile(2, 2, 1)
        texcoords = (texcoords % 1.0) * 0.5
        texcoords = torch.where(texcoords >= 0.25, texcoords, texcoords + 0.5)
        wrap = "reflection"
    texcoords = (texcoords * 2 - 1) * flipper
    return F.grid_sample(
        texture2d[None].permute(0, 3, 1, 2),
        texcoords,
        padding_mode=wrap, mode=mode, align_corners=align_corners
    ).view(texture2d.shape[-1], -1).T


def sample2d(
    texture2d: torch.Tensor,
    texcoords: torch.Tensor,
    wrap: str = "border",
    mode: str = "bilinear"
):
    """
    Samples a 2D texture on UV coords.

    Args:
        texture2d (torch.Tensor):
            Tensor of shape (H, W, C), the texture to be sampled.
        texcoords (torch.Tensor):
            Tensor of shape (..., 2), batch of coordinates to sample.
            (0, 0) is the bottom-left corner (last row in H axis).
        wrap (str):
            Defines how the boundaries and points outside boundaries are handled.
            One of "border", "reflection" and "cyclic".
        mode (str):
            Sampling method.
            One of "nearest", "bilinear" and "bicubic".
    
    Returns:
        torch.Tensor: Tensor of shape (..., C), sampled texture.
            The batch dimensions are kept the same as ``texcoords``.
    """
    # bhwc -> bchw
    # texcoords: ..., 2
    original_shape = texcoords.shape
    sampled: torch.Tensor = _sample2d_internal(texture2d, texcoords, flipper_2d(), wrap, mode)
    # bchw -> wc -> ..., c
    return sampled.reshape(*original_shape[:-1], texture2d.shape[-1])


@torch.jit.script
def _sample3d_internal(
    texture3d: torch.Tensor,
    texcoords: torch.Tensor,
    flipper: torch.Tensor,
    wrap: str = "border",
    mode: str = "bilinear"
):
    texcoords = texcoords.reshape(1, 1, 1, -1, texcoords.shape[-1])
    texcoords = (texcoords * 2 - 1) * flipper
    align_corners = False
    if wrap == "latlong":
        align_corners = True
        wrap = "reflection"
    return F.grid_sample(
        texture3d[None].permute(0, 4, 1, 2, 3),
        texcoords,
        padding_mode=wrap, mode=mode, align_corners=align_corners
    ).view(texture3d.shape[-1], -1).T


def sample3d(
    texture3d: torch.Tensor,
    texcoords: torch.Tensor,
    wrap: str = "border",
    mode: str = "bilinear"
):
    """
    Samples a 3D texture on UV coords.

    Args:
        texture3d (torch.Tensor):
            Tensor of shape (D, H, W, C), the texture to be sampled.
        texcoords (torch.Tensor):
            Tensor of shape (..., 3), batch of coordinates to sample.
            (0, 0, 0) is the bottom-left corner, and the first slice of the depth.
        wrap (str):
            Defines how the boundaries and points outside boundaries are handled.
            One of "border" and "reflection".
        mode (str):
            Sampling method.
            One of "nearest" and "bilinear".
    
    Returns:
        torch.Tensor: Tensor of shape (..., C), sampled texture.
            The batch dimensions are kept the same as ``texcoords``.
    """
    # bdhwc -> bcdhw
    # texcoords: ..., 2
    original_shape = texcoords.shape
    sampled: torch.Tensor = _sample3d_internal(texture3d, texcoords, flipper_3d(), wrap, mode)
    # bcdhw -> wc -> ..., c
    return sampled.reshape(*original_shape[:-1], texture3d.shape[-1])


def reflect(i: torch.Tensor, n: torch.Tensor):
    """
    Reflects a ray according to a normal vector.

    The method is broadcastable.

    Args:
        i: Inbound ray. Shape (..., 3).
        n: Normal. Shape (..., 3).
    Returns:
        torch.Tensor: Tensor of the same shape as inputs.
    """
    return fma(dot(n, i), n, i, -2.0)


def saturate(x: torch.Tensor):
    """
    Clamps values in ``x`` to be between 0 and 1.

    The method is element-wise.
    """
    return torch.clamp(x, 0.0, 1.0)


def split_alpha(x: torch.Tensor):
    """
    Splits ``x`` on the last dimension.
    The singleton alpha dimension is kept.

    The method is broadcastable.

    Args:
        x (torch.Tensor): Shape (..., C + 1).
    Returns:
        Tuple of tensors (c, alpha):
            Shapes (..., C) and (..., 1), respectively.
    """
    return x[..., :-1], x[..., -1:]


def normalized(x: torch.Tensor):
    """
    Normalize ``x`` vector over the last dimension.
    Zero vectors are kept as-is.

    The method is broadcastable.
    """
    return F.normalize(x, dim=-1)


def length(x: torch.Tensor):
    """
    Compute the length of ``x`` vector over the last dimension.

    The method is broadcastable.

    Args:
        x (torch.Tensor): Shape (..., C).
    Returns:
        torch.Tensor: Shape (..., 1), L2 length of vectors.
    """
    return torch.norm(x, dim=-1, keepdim=True)


def floatx(*tensors):
    """
    Make a new tensor from sub-tensors or floats.

    Inputs are broadcast and concatenated on the last dimension.

    Example: floatx(rgba.rgb, 1) can give a new RGBA tensor with alpha set to opaque.
    """
    tensors = [x if isinstance(x, torch.Tensor) else gpu_f32(x) for x in tensors]
    ref_shape = tensors[0].shape[:-1]
    if all(x.shape[:-1] == ref_shape for x in tensors[1:]):
        return torch.cat(tensors, dim=-1)
    tensors = rst.supercat(tensors, dim=-1)
    return tensors


def float4(*tensors):
    """
    See ``floatx``. The result should have 4 as the last channel dimension.
    """
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 4
    return tensors


def float3(*tensors):
    """
    See ``floatx``. The result should have 3 as the last channel dimension.
    """
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 3
    return tensors


def float2(*tensors):
    """
    See ``floatx``. The result should have 2 as the last channel dimension.
    """
    tensors = floatx(*tensors)
    assert tensors.shape[-1] == 2
    return tensors


def to_bchw(hwc: torch.Tensor):
    """
    Convert a HWC or BHWC tensor into BCHW format.
    """
    if hwc.ndim == 3:
        hwc = hwc[None]
    return hwc.permute(0, 3, 1, 2)


def to_c1hw(hwc: torch.Tensor):
    """
    Convert a HWC tensor into C1HW format, i.e. a batch of single-channel images in BCHW format.
    """
    return hwc.permute(2, 0, 1)[:, None]


def from_c1hw(c1hw: torch.Tensor):
    """
    Convert a C1HW tensor back to HWC format.
    """
    return c1hw.squeeze(1).permute(1, 2, 0)


def to_hwc(bchw: torch.Tensor):
    """
    Convert a BCHW or CHW tensor back to HWC format.
    """
    if bchw.ndim == 3:
        bchw = bchw[None]
    return bchw.permute(0, 2, 3, 1).squeeze(0)


def blur2d(texture2d: torch.Tensor, size: int, iters: int):
    """
    Blur an 2D image (HWC) by the box kernel for several iterations,
    efficiently simulating a Gaussian-like blur.

    Args:
        texture2d: HWC tensor to blur.
        size: Size of box kernel.
        iters: Iterations of box blur.
    Returns:
        torch.Tensor: Has the same shape as ``texture2d``.
    """
    tex = to_c1hw(texture2d)
    ks = 2 * size + 1
    ker = tex.new_ones([1, 1, ks, ks]) / ks ** 2
    for _ in range(iters):
        tex = torch.conv2d(tex, ker, padding=size)
    return from_c1hw(tex)


def sobel(texture2d: torch.Tensor):
    """
    Compute the Sobel edge detection operator on a grayscale image.

    Args:
        texture2d (torch.Tensor): (H, W, 1) grayscale image.
    Returns:
        torch.Tensor: (H, W, 4) containing (strength_x, strength_y, strength_total, input).
    """
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
    """
    Convert RGB to HSV. Broadcastable.
    """
    return _rgb2hsv_internal(rgb.flatten(0, -2)).reshape(rgb.shape)


def hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV to RGB. Broadcastable.
    """
    return _hsv2rgb_internal(hsv.flatten(0, -2)).reshape(hsv.shape)


def _rgb2hsv_internal(rgb: torch.Tensor) -> torch.Tensor:
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


def _hsv2rgb_internal(hsv: torch.Tensor) -> torch.Tensor:
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


@singleton_cached
def white_tex():
    """
    Get a full white (ones) RGBA image of shape [16, 16, 4].
    """
    return gpu_f32(numpy.ones([16, 16, 4], dtype=numpy.float32))


@singleton_cached
def black_tex():
    """
    Get a full black (zeros) RGBA image of shape [16, 16, 4].
    """
    return gpu_f32(numpy.zeros([16, 16, 4], dtype=numpy.float32))


@singleton_cached
def gray_tex():
    """
    Get a full gray (0.5) RGBA image of shape [16, 16, 4].
    """
    return gpu_f32(numpy.full([16, 16, 4], 0.5, dtype=numpy.float32))


@singleton_cached
def empty_normal_tex():
    """
    Get a full empty ([0.5, 0.5, 1.0]) normal image of shape [16, 16, 3].
    """
    return gpu_f32(numpy.full([16, 16, 3], [0.5, 0.5, 1.0], dtype=numpy.float32))


class _SmallMatrixInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor):
        output = matrix.new_tensor(numpy.linalg.inv(matrix.detach().cpu().numpy()))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor):
        output, = ctx.saved_tensors
        output = output.transpose(-1, -2)
        return -output @ grad_output @ output


def small_matrix_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Faster inverse for matrices smaller than 2x8x8.
    """
    return _SmallMatrixInverse.apply(x)


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


if hasattr(torch.linalg, 'vecdot'):
    def dot(a: torch.Tensor, b: torch.Tensor, keepdim: bool = True):
        if keepdim:
            return torch.linalg.vecdot(a, b).unsqueeze(-1)
        else:
            return torch.linalg.vecdot(a, b)
else:
    def dot(a: torch.Tensor, b: torch.Tensor, keepdim: bool = True):
        return (a * b).sum(-1, keepdim=keepdim)

if hasattr(torch.linalg, 'cross'):
    def cross(a: torch.Tensor, b: torch.Tensor):
        return torch.linalg.cross(a, b)
else:
    def cross(a: torch.Tensor, b: torch.Tensor):
        return torch.cross(a, b, dim=-1)


for _idx in ['xyzw', 'rgba']:
    _comb = [''.join(x) for x in itertools.product([''] + list(_idx), repeat=len(_idx))]
    _comb = [x for x in _comb if x]
    for attr in _comb:
        _make_attr(attr, _idx)
