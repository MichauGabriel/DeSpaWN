# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring of High-Frequency Time Series
------          (DeSpaWN)
Description: 
--------------
Toy script to showcase the deep neural network DeSpaWN.
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

from lib import despawnLayers as impLay


class DeSpaWN(nn.Module):
    """Learnable multilevel wavelet analysis and reconstruction network.

    Inputs use the shape ``(batch, time, 1, 1)``. The forward pass returns
    the reconstruction, coefficient loss, final low-pass coefficients, and
    high-pass coefficients ordered from coarse to fine scale.
    """

    def __init__(self, kernelInit, kernTrainable, level, lossCoeff,
                 kernelsConstraint, initHT, trainHT):
        """Build the wavelet filters and threshold modules.

        Args:
            kernelInit: Integer kernel length or initial kernel values.
            kernTrainable: Whether wavelet kernels receive gradients.
            level: Number of analysis and reconstruction levels.
            lossCoeff: Coefficient loss mode, either ``'l1'`` or ``None``.
            kernelsConstraint: Filter sharing mode: ``'CQF'``,
                ``'PerLayer'``, ``'PerFilter'``, or ``'Free'``.
            initHT: Initial positive and negative threshold values.
            trainHT: Whether threshold values receive gradients.
        """
        super().__init__()
        self.level = level
        self.lossCoeff = lossCoeff

        if kernelsConstraint == 'CQF':
            kernel = impLay.Kernel(kernelInit, kernTrainable)
            kernelsG = nn.ModuleList([kernel for _ in range(level)])
            kernelsH = kernelsG
            kernelsGT = kernelsG
            kernelsHT = kernelsG
        elif kernelsConstraint == 'PerLayer':
            kernelsG = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsH = kernelsG
            kernelsGT = kernelsG
            kernelsHT = kernelsG
        elif kernelsConstraint == 'PerFilter':
            kernelsG = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsH = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsGT = kernelsG
            kernelsHT = kernelsH
        elif kernelsConstraint == 'Free':
            kernelsG = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsH = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsGT = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
            kernelsHT = nn.ModuleList([impLay.Kernel(kernelInit, kernTrainable) for _ in range(level)])
        else:
            raise ValueError('Could not understand value in kernelsConstraint')

        self.kernelsG = kernelsG
        self.kernelsH = kernelsH
        self.kernelsGT = kernelsGT
        self.kernelsHT = kernelsHT
        self.thresholds = nn.ModuleList([
            impLay.HardThresholdAssym(init=initHT, trainBias=trainHT)
            for _ in range(level + 1)
        ])
        self.lowPassWave = impLay.LowPassWave()
        self.highPassWave = impLay.HighPassWave()
        self.lowPassTrans = impLay.LowPassTrans()
        self.highPassTrans = impLay.HighPassTrans()

    def forward(self, inputSig):
        """Decompose, threshold, and reconstruct a batch of signals.

        Args:
            inputSig: Tensor shaped ``(batch, time, 1, 1)``.

        Returns:
            A tuple ``(reconstruction, coefficientLoss, lowPass, *highPass)``.
            The high-pass tensors are returned from the coarsest to the
            finest decomposition level.
        """
        g = inputSig
        highPass = []
        inputSizes = []

        for levelIndex in range(self.level):
            inputSizes.append(g.shape)
            highPass.append(self.thresholds[levelIndex](
                self.highPassWave((g, self.kernelsH[levelIndex]())))
            )
            g = self.lowPassWave((g, self.kernelsG[levelIndex]()))

        g = self.thresholds[self.level](g)
        gint = g

        for levelIndex in range(self.level - 1, -1, -1):
            high = self.highPassTrans((
                highPass[levelIndex], self.kernelsHT[levelIndex](), inputSizes[levelIndex]))
            g = self.lowPassTrans((g, self.kernelsGT[levelIndex](), inputSizes[levelIndex]))
            g = g + high

        if self.lossCoeff is None:
            coeffLoss = torch.zeros((1, 1, 1, 1), dtype=inputSig.dtype, device=inputSig.device)
        elif self.lossCoeff == 'l1':
            coeffLoss = torch.mean(torch.abs(torch.cat([gint, *highPass], dim=1)), dim=1, keepdim=True)
        else:
            raise ValueError("Could not understand value in 'lossCoeff'. It should be either 'l1' or 'None'")

        return g, coeffLoss, gint, *highPass[::-1]