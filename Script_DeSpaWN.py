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

import tensorflow as tf # designed with 2.1.0 /!\ models output changes with tf>2.2.0
import tensorflow.keras as keras

from lib import despawn


# Load a toy time series data to run DeSPAWN
signal = pd.read_csv("monthly-sunspots.csv")
lTrain = 2000 # length of the training section
signalT = ((signal['Sunspots']-signal['Sunspots'].mean())/signal['Sunspots'].std()).values[np.newaxis,:,np.newaxis,np.newaxis]
signal = signalT[:,:lTrain,:,:]

# Number of decomposition level is max log2 of input TS
level = np.floor(np.log2(signal.shape[1])).astype(int)
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


epochs = 1000
verbose = 2

# Set sparsity (dummy) loss:
def coeffLoss(yTrue,yPred):
    return lossFactor*tf.reduce_mean(yPred,keepdims=True)
# Set residual loss:
def recLoss(yTrue,yPred):
    return tf.math.abs(yTrue-yPred)

keras.backend.clear_session()
# generates two models: 
#      model1 outputs the reconstructed signals and the loss on the wavelet coefficients
#      model2 outputs the reconstructed signals and wavelet coefficients
model1,model2 = despawn.createDeSpaWN(inputSize=None, kernelInit=kernelInit, kernTrainable=kernTrainable, level=level, lossCoeff=lossCoeff, kernelsConstraint=mode, initHT=initHT, trainHT=trainHT)
opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
# For the training we only use model1
model1.compile(optimizer=opt, loss=[recLoss, coeffLoss])
# the sparsity term has no ground truth => just input an empty numpy array as ground truth (anything would do, in coeffLoss, yTrue is not called)
H = model1.fit(signal,[signal,np.empty((signal.shape[0]))], epochs=epochs, verbose=verbose)

# Examples for plotting the model outputs and learnings
indPlot = 0
out  = model1.predict(signal)
outC = model2.predict(signal)
# Test part of the signal
outTe  = model1.predict(signalT[:,lTrain:,:,:])
outCTe = model2.predict(signalT[:,lTrain:,:,:])

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(2,1,1)
ax.plot(np.arange(signal.shape[1]),signal[indPlot,:,0,0])
ax.plot(np.arange(signal.shape[1]),out[0][indPlot,:,0,0])
ax.plot(np.arange(signal.shape[1],signalT.shape[1]),signalT[indPlot,lTrain:,0,0])
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
