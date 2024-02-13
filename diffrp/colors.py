import torch


def linear_to_srgb(rgb):
    if torch.is_grad_enabled():
        rgb = torch.clamp_min(rgb, 1e-5)
    return torch.where(rgb < 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def srgb_to_linear(rgb):
    if torch.is_grad_enabled():
        rgb = torch.clamp_min(rgb, 1e-5)
    return torch.where(rgb < 0.04045, rgb * (1 / 12.92), (1 / 1.055) * ((rgb + 0.055) ** 2.4))


def aces_simple(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def aces_fit(color: torch.Tensor):
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
