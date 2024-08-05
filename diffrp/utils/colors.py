import torch


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts from linear space to sRGB space.
    It is recommended to have the inputs already in LDR (range 0 to 1).

    The output has the same shape as the input.
    """
    if torch.is_grad_enabled():
        rgb = torch.clamp_min(rgb, 1e-5)
    return torch.where(rgb < 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts from sRGB space to linear space.

    The output has the same shape as the input.
    """
    if torch.is_grad_enabled():
        rgb = torch.clamp_min(rgb, 1e-5)
    return torch.where(rgb < 0.04045, rgb * (1 / 12.92), (1 / 1.055) * ((rgb + 0.055) ** 2.4))


def aces_simple(x):
    """
    Simple ACES mapping of color values.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def aces_fit(color: torch.Tensor):
    """
    More accurate ACES mapping of color values.
    """
    aces_i = color.new_tensor([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])
    aces_o = color.new_tensor([
        [ 1.60475, -0.53108, -0.07367],
        [-0.10208,  1.10813, -0.00605],
        [-0.00327, -0.07276,  1.07602]
    ])

    v = color.matmul(aces_i.T)
    
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    v = a / b

    v = v.matmul(aces_o.T)

    return torch.clamp(v, 0, 1)


def linear_to_alexa_logc_ei1000(x):
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    return torch.where(x > cut, c * torch.log10(a * x + b) + d, e * x + f)


def alexa_logc_ei1000_to_linear(logc):
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    return torch.where(logc > e * cut + f, (10 ** ((logc - d) / c) - b) / a, (logc - f) / e)
