# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import torch.nn as nn
from .win_attention import WinBasedAttention

__all__ = [
    "conv3x3",
    "conv3x3_adp",
    "subpel_conv3x3",
    "subpel_conv3x3_adp",
    "conv1x1",
    "Win_noShift_Attention",
    "Conv3x3Adapter",
]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class Conv3x3Adapter():
    def __init__(self, in_ch, out_ch, adp_ch,):
        

        self.down_project = conv3x3(in_ch, adp_ch, stride)
        self.up_project = conv3x3(adp_ch, out_ch, stride)
    
    def forward(self, x):
        x = self.down_project(x)
        x = self.up_project(x)
        return x

class Conv3x3WithAdp():
    def __init__(self, in_ch, out_ch, alpha, adp_ch, stride=1):
        self.conv = conv3x3(in_ch, out_ch, stride)
        self.adp = Conv3x3Adapter(in_ch, out_ch, adp_ch, stride)
        self.alpha = alpha
        
    def forward(self, x):
        return self.conv(x) + self.alpha * self.adp(x)


def conv3x3_adp(in_ch: int, out_ch: int, alpha: float, adp_ch: int, stride: int = 1) -> nn.Module:
    return Conv3x3WithAdp(in_ch, out_ch, alpha, adp_ch, stride)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def subpel_conv3x3_adp(in_ch: int, out_ch: int, alpha: int, adp_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        conv3x3_adp(in_ch, out_ch * r ** 2, alpha, adp_ch), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

