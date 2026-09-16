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
# Usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib import despawn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Load a toy time series data to run DeSPAWN
signal = pd.read_csv("monthly-sunspots.csv")
lTrain = 2000 # length of the training section
signalT = ((signal['Sunspots']-signal['Sunspots'].mean())/signal['Sunspots'].std()).values.copy()[np.newaxis,:]
signalT = torch.from_numpy(signalT).float()
signal = signalT[:,:lTrain]
signal = torch.cat([signal,signal],dim=0)
# Train on fixed-size windows. Windows never cross the boundary between
# independent time series; samples left over at the end of each series are
# kept for full-length evaluation but are not used for training.
windowSize = 2000
batchSize = 1
signal = torch.as_tensor(signal, dtype=torch.float32)
if signal.ndim != 2:
    raise ValueError('signal must have shape (number_of_measurements, time_series_length)')
if windowSize > signal.shape[1]:
    raise ValueError('windowSize must not exceed the training time-series length')
trainingWindows = signal.unfold(1, windowSize, windowSize)
trainingWindows = trainingWindows.contiguous().reshape(-1, windowSize, 1, 1)
trainLoader = DataLoader(TensorDataset(trainingWindows), batch_size=batchSize,
                         shuffle=True, pin_memory=device.type == 'cuda')

# Number of decomposition levels is based on the training window size.
level = 10
# Train hard thresholding (HT) coefficient?
trainHT = True
# Initialise HT value
initHT = 0.3
# Which loss to consider for wavelet coeffs ('l1' or None)
lossCoeff='l1'
# Weight for sparsity loss versus residual?
lossFactor = 1.0
# Train wavelets? (Trainable kernels)
kernTrainable = True
# Which training mode?
# cf (https://arxiv.org/pdf/2105.00899.pdf -- https://doi.org/10.1073/pnas.2106598119) [Section 4.4 Ablation Study]
#   CQF => learn wavelet 0 infer all other kernels from the network
#   PerLayer => learn one wavelet per level, infer others
#   PerFilter => learn wavelet + scaling function per level + infer other
#   Free => learn everything
mode = 'PerLayer' # QMF PerLayer PerFilter Free

# Initialise wavelet kernel (here db-4)
kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                           -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])


epochs = 460
verbose = 2

model = despawn.DeSpaWN(kernelInit=kernelInit, kernTrainable=kernTrainable,
                        level=level, lossCoeff=lossCoeff,
                        kernelsConstraint=mode, initHT=initHT,
                        trainHT=trainHT).to(device)
opt = torch.optim.NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)
model.train()
H = []
for epoch in range(epochs):
    opt.zero_grad()
    epochLoss = 0.0
    epochSamples = 0
    for (batch,) in trainLoader:
        batch = batch.to(device, non_blocking=True)
        outputs = model(batch)
        reconstruction, coeff = outputs[:2]
        loss = torch.mean(torch.abs(batch-reconstruction)) + lossFactor*torch.mean(coeff)
        loss.backward()
        opt.step()
        epochLoss += loss.item() * batch.shape[0]
        epochSamples += batch.shape[0]
        opt.zero_grad()
    H.append(epochLoss / epochSamples)
    if verbose == 2:
        print(f'{epoch + 1}/{epochs} - loss: {H[-1]:.6f}')

# Examples for plotting the model outputs and learnings
indPlot = 0
model.eval()
signalInput = signalT[:,:lTrain]
signalTest = signalT[:,lTrain:]
#signalTestInput = signalTest.unsqueeze(-1).unsqueeze(-1)
with torch.no_grad():
    outputs = model(signalInput.unsqueeze(-1).unsqueeze(-1).to(device))
out = tuple(value.detach().cpu().numpy() for value in outputs[:2])
outC = tuple(value.detach().cpu().numpy() for value in (outputs[0], outputs[2], *outputs[3:]))
# Test part of the signal
with torch.no_grad():
    outputsTe = model(signalTest.unsqueeze(-1).unsqueeze(-1).to(device))
outTe = tuple(value.detach().cpu().numpy() for value in outputsTe[:2])
outCTe = tuple(value.detach().cpu().numpy() for value in (outputsTe[0], outputsTe[2], *outputsTe[3:]))

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(2,1,1)
ax.plot(np.arange(signal.shape[1]),signal[indPlot])
ax.plot(np.arange(signal.shape[1]),out[0][indPlot,:,0,0])
ax.plot(np.arange(signal.shape[1],signalT.shape[1]),signalTest[indPlot])
ax.plot(np.arange(signal.shape[1],signalT.shape[1]),outTe[0][indPlot,:,0,0])
ax.legend(['Train Original','Train Reconstructed','Test Original', 'Test Reconstructed'])
ax = fig.add_subplot(2,2,3)
idpl = 0
for e,o in enumerate(outC[1:]):
    ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:])), positions=[e], widths=0.8)
ax.set_xlabel('Decomposition Level')
ax.set_ylabel('Coefficient Distribution')
trainYLim = ax.get_ylim()
trainXLim = ax.get_xlim()
ax = fig.add_subplot(2,2,4)
idpl = 0
for e,o in enumerate(outCTe[1:]):
    print(o.shape[1])
    if o.shape[1]>1:
        ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:])), positions=[e], widths=0.8)
    else:
        ax.plot(e,np.abs(np.squeeze(o[indPlot,:,:,:])),'o',color='k')
ax.set_xlabel('Decomposition Level')
ax.set_ylabel('Coefficient Distribution')
ax.set_ylim(trainYLim)
ax.set_xlim(trainXLim)
