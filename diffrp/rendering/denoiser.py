"""
Experimental. Implementation and interface subject to change.
"""
## Copied and modified from intel OIDN
## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..resources import get_resource_path
from ..utils import *


def get_denoiser():
    net = UNet(9, 3)
    net.load_state_dict(torch_load_no_warning(get_resource_path("denoisers/rt_hdr_alb_nrm.pt")))
    if torch.cuda.is_available():
        net = net.cuda()
    return torch.jit.script(net.float())


def run_denoiser(
    denoiser, pbr_hdr: torch.Tensor, albedo_srgb: torch.Tensor, normal: torch.Tensor,
    alignment: int = 16
):
    PU_NORM_SCALE = 1. / linear_to_pu(torch.tensor(65504)).item()
    in_img = to_bchw(torch.cat([linear_to_pu(pbr_hdr) * PU_NORM_SCALE, albedo_srgb, normal * 0.5 + 0.5], dim=-1))
    b, c, h, w = in_img.shape
    dh = math.ceil(h / alignment) * alignment - h
    dw = math.ceil(w / alignment) * alignment - w
    in_img = nn.ReflectionPad2d((dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))(in_img)
    b, c, h, w = in_img.shape
    return pu_to_linear(to_hwc(denoiser(in_img))[dh // 2: h - (dh - dh // 2), dw // 2: w - (dw - dw // 2)] / PU_NORM_SCALE)


## -----------------------------------------------------------------------------
## Network layers
## -----------------------------------------------------------------------------


# 3x3 convolution module
def Conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


# ReLU function
def relu(x):
    return F.relu(x, inplace=True)


# 2x2 max pool function
def pool(x):
    return F.max_pool2d(x, 2, 2)


# 2x2 nearest-neighbor upsample function
def upsample(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


# Channel concatenation function
def concat(a, b):
    return torch.cat((a, b), 1)


## -----------------------------------------------------------------------------
## U-Net model
## -----------------------------------------------------------------------------


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, small=False):
        super(UNet, self).__init__()

        # Number of channels per layer
        ic = in_channels
        if small:
            ec1 = 32
            ec2 = 32
            ec3 = 32
            ec4 = 32
            ec5 = 32
            dc4 = 64
            dc3 = 64
            dc2a = 64
            dc2b = 32
            dc1a = 32
            dc1b = 32
        else:
            ec1 = 32
            ec2 = 48
            ec3 = 64
            ec4 = 80
            ec5 = 96
            dc4 = 112
            dc3 = 96
            dc2a = 64
            dc2b = 64
            dc1a = 64
            dc1b = 32
        oc = out_channels

        # Convolutions
        self.enc_conv0 = Conv(ic, ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5a = Conv(ec4, ec5)
        self.enc_conv5b = Conv(ec5, ec5)
        self.dec_conv4a = Conv(ec5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2a)
        self.dec_conv2b = Conv(dc2a, dc2b)
        self.dec_conv1a = Conv(dc2b + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc)

        # Images must be padded to multiples of the alignment
        self.alignment = 16

    def forward(self, input):
        # Encoder
        # -------------------------------------------

        x = relu(self.enc_conv0(input))  # enc_conv0

        x = relu(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1

        x = relu(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2

        x = relu(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3

        x = relu(self.enc_conv4(x))  # enc_conv4
        x = pool(x)  # pool4

        # Bottleneck
        x = relu(self.enc_conv5a(x))  # enc_conv5a
        x = relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = relu(self.dec_conv4a(x))  # dec_conv4a
        x = relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = relu(self.dec_conv3a(x))  # dec_conv3a
        x = relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = relu(self.dec_conv2a(x))  # dec_conv2a
        x = relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, input)  # concat1
        x = relu(self.dec_conv1a(x))  # dec_conv1a
        x = relu(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x


## -----------------------------------------------------------------------------
## U-Net model: large
## -----------------------------------------------------------------------------


class UNetLarge(nn.Module):
    def __init__(self, in_channels, out_channels, xl=False):
        super(UNetLarge, self).__init__()

        # Number of channels per layer
        ic = in_channels
        if xl:
            ec1 = 96
            ec2 = 128
            ec3 = 192
            ec4 = 256
            ec5 = 384
            dc4 = 256
            dc3 = 192
            dc2 = 128
            dc1 = 96
        else:
            ec1 = 64
            ec2 = 96
            ec3 = 128
            ec4 = 192
            ec5 = 256
            dc4 = 192
            dc3 = 128
            dc2 = 96
            dc1 = 64
        oc = out_channels

        # Convolutions
        self.enc_conv1a = Conv(ic, ec1)
        self.enc_conv1b = Conv(ec1, ec1)
        self.enc_conv2a = Conv(ec1, ec2)
        self.enc_conv2b = Conv(ec2, ec2)
        self.enc_conv3a = Conv(ec2, ec3)
        self.enc_conv3b = Conv(ec3, ec3)
        self.enc_conv4a = Conv(ec3, ec4)
        self.enc_conv4b = Conv(ec4, ec4)
        self.enc_conv5a = Conv(ec4, ec5)
        self.enc_conv5b = Conv(ec5, ec5)
        self.dec_conv4a = Conv(ec5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2)
        self.dec_conv2b = Conv(dc2, dc2)
        self.dec_conv1a = Conv(dc2 + ic, dc1)
        self.dec_conv1b = Conv(dc1, dc1)
        self.dec_conv1c = Conv(dc1, oc)

        # Images must be padded to multiples of the alignment
        self.alignment = 16

    def forward(self, input):
        # Encoder
        # -------------------------------------------

        x = relu(self.enc_conv1a(input))  # enc_conv1a
        x = relu(self.enc_conv1b(x))  # enc_conv1b
        x = pool1 = pool(x)  # pool1

        x = relu(self.enc_conv2a(x))  # enc_conv2a
        x = relu(self.enc_conv2b(x))  # enc_conv2b
        x = pool2 = pool(x)  # pool2

        x = relu(self.enc_conv3a(x))  # enc_conv3a
        x = relu(self.enc_conv3b(x))  # enc_conv3b
        x = pool3 = pool(x)  # pool3

        x = relu(self.enc_conv4a(x))  # enc_conv4a
        x = relu(self.enc_conv4b(x))  # enc_conv4b
        x = pool(x)  # pool4

        # Bottleneck
        x = relu(self.enc_conv5a(x))  # enc_conv5a
        x = relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = relu(self.dec_conv4a(x))  # dec_conv4a
        x = relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = relu(self.dec_conv3a(x))  # dec_conv3a
        x = relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = relu(self.dec_conv2a(x))  # dec_conv2a
        x = relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, input)  # concat1
        x = relu(self.dec_conv1a(x))  # dec_conv1a
        x = relu(self.dec_conv1b(x))  # dec_conv1b
        x = relu(self.dec_conv1c(x))  # dec_conv1c

        return x
