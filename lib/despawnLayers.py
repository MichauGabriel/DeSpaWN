# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring of High-Frequency Time Series
------          (DeSpaWN)

Description: 
--------------
Function to generate the layers used in DeSpaWN TF model.
Please cite the corresponding paper:
Michau, G., Frusque, G., & Fink, O. (2022).
Fully learnable deep wavelet transform for unsupervised monitoring of high-frequency time series.
Proceedings of the National Academy of Sciences, 119(8).

Version: 1.0
--------

@author:  Dr. Gabriel Michau,
--------  Chair of Intelligent Maintenance Systems
          ETH Zürich

Created on 15.01.2022

Licence:
----------
MIT License

Copyright (c) 2022 Dr. Gabriel Michau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import torch
from torch import nn
from torch.nn import functional as F


class Kernel(nn.Module):
    """Trainable one-dimensional wavelet kernel stored for 2D convolutions."""

    def __init__(self, kernelInit=8, trainKern=True, **kwargs):
        """Initialize a kernel from a length or an array of coefficients.

        Args:
            kernelInit: Integer kernel length for random initialization, or
                an array-like collection of initial coefficients.
            trainKern: Whether the kernel is trainable.
        """
        super().__init__()
        if isinstance(kernelInit, int):
            kernel = torch.randn(kernelInit, 1, 1, 1)
        else:
            kernel = torch.as_tensor(kernelInit, dtype=torch.float32).reshape(-1, 1, 1, 1)
        self.kernel = nn.Parameter(kernel, requires_grad=trainKern)

    def forward(self, inputs=None):
        """Return the kernel coefficients."""
        return self.kernel


def _to_channels_first(signal):
    """Convert ``(batch, time, height, channels)`` to PyTorch layout."""
    return signal.permute(0, 3, 1, 2)


def _to_channels_last(signal):
    """Convert PyTorch layout back to ``(batch, time, height, channels)``."""
    return signal.permute(0, 2, 3, 1)


def _same_padding(length, kernelSize, stride=2):
    """Return asymmetric padding that gives ceil(length / stride) output."""
    outputLength = (length + stride - 1) // stride
    padding = max((outputLength - 1) * stride + kernelSize - length, 0)
    return padding // 2, padding - padding // 2


class LowPassWave(nn.Module):
    """Apply a stride-two low-pass analysis filter."""

    def forward(self, inputs):
        """Filter and downsample a signal along its time dimension."""
        signal, kernel = inputs
        signal = _to_channels_first(signal)
        kernel = kernel.permute(3, 2, 0, 1)
        padBefore, padAfter = _same_padding(signal.shape[2], kernel.shape[2])
        signal = F.pad(signal, (0, 0, padBefore, padAfter))
        result = F.conv2d(signal, kernel, stride=(2, 1))
        return _to_channels_last(result)


class HighPassWave(nn.Module):
    """Apply the quadrature-mirror high-pass analysis filter."""

    def forward(self, inputs):
        """Create the alternating high-pass filter and downsample."""
        signal, kernel = inputs
        mask = torch.pow(-1.0, torch.arange(kernel.shape[0], device=kernel.device, dtype=kernel.dtype))
        kernel = torch.flip(kernel, dims=(0,)) * mask.reshape(-1, 1, 1, 1)
        return LowPassWave()((signal, kernel))


class LowPassTrans(nn.Module):
    """Apply a stride-two low-pass synthesis filter."""

    def forward(self, inputs):
        """Upsample and reconstruct a low-pass signal to its input length."""
        signal, kernel, inputSize = inputs
        signal = _to_channels_first(signal)
        kernel = kernel.permute(3, 2, 0, 1)
        raw = F.conv_transpose2d(signal, kernel, stride=(2, 1))
        padBefore, padAfter = _same_padding(inputSize[1], kernel.shape[2])
        raw = raw[:, :, padBefore:padBefore + inputSize[1], :]
        return _to_channels_last(raw)


class HighPassTrans(nn.Module):
    """Apply the quadrature-mirror high-pass synthesis filter."""

    def forward(self, inputs):
        """Create the alternating high-pass filter and upsample."""
        signal, kernel, inputSize = inputs
        mask = torch.pow(-1.0, torch.arange(kernel.shape[0], device=kernel.device, dtype=kernel.dtype))
        kernel = torch.flip(kernel, dims=(0,)) * mask.reshape(-1, 1, 1, 1)
        return LowPassTrans()((signal, kernel, inputSize))


class HardThresholdAssym(nn.Module):
    """Differentiable asymmetric threshold for positive and negative values."""

    def __init__(self, init=None, trainBias=True, **kwargs):
        """Initialize independent positive and negative thresholds.

        Args:
            init: Initial value for both thresholds; defaults to ``1.0``.
            trainBias: Whether the thresholds are trainable.
        """
        super().__init__()
        value = 1.0 if init is None else float(init)
        self.thrP = nn.Parameter(torch.full((1, 1, 1, 1), value), requires_grad=trainBias)
        self.thrN = nn.Parameter(torch.full((1, 1, 1, 1), value), requires_grad=trainBias)

    def forward(self, inputs):
        """Suppress values near zero using smooth sigmoid gates."""
        return inputs * (torch.sigmoid(10 * (inputs - self.thrP)) +
                         torch.sigmoid(-10 * (inputs + self.thrN)))
