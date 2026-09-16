"""PyTorch implementation of the DeSpaWN wavelet network."""

import torch
from torch import nn

from lib import despawnLayers as impLay


class _DeSpaWNCore(nn.Module):
    def __init__(self, kernelInit, kernTrainable, level, lossCoeff,
                 kernelsConstraint, initHT, trainHT):
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

        return g, coeffLoss, gint, highPass[::-1]


class _DeSpaWNOutput(nn.Module):
    def __init__(self, core, includeCoefficients):
        super().__init__()
        self.core = core
        self.includeCoefficients = includeCoefficients

    def forward(self, inputSig):
        reconstruction, coeffLoss, gint, highPass = self.core(inputSig)
        if self.includeCoefficients:
            return reconstruction, gint, *highPass
        return reconstruction, coeffLoss


def createDeSpaWN(inputSize=None, kernelInit=8, kernTrainable=True, level=1,
                  lossCoeff='l1', kernelsConstraint='QMF', initHT=1.0,
                  trainHT=True):
    """Create two PyTorch models sharing the same DeSpaWN parameters.

    The first model returns reconstruction and coefficient loss. The second
    returns reconstruction followed by the low-pass and high-pass outputs.
    """
    del inputSize
    core = _DeSpaWNCore(kernelInit, kernTrainable, level, lossCoeff,
                        kernelsConstraint, initHT, trainHT)
    return _DeSpaWNOutput(core, False), _DeSpaWNOutput(core, True)