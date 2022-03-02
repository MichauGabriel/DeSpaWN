# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring of High-Frequency Time Series
------          (DeSpaWN)

Description: 
--------------
Function to generate a DeSpaWN TF model.
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
# /!\ Designed for tensorflow 2.1.X
import tensorflow as tf
import tensorflow.keras as keras
from lib import despawnLayers as impLay

def createDeSpaWN(inputSize=None, kernelInit=8, kernTrainable=True, level=1, lossCoeff='l1', kernelsConstraint='QMF', initHT=1.0, trainHT=True):
    """
    Function that generates a TF DeSpaWN network

    Parameters
    ----------
    inputSize : INT, optional
        Length of the time series. Network is more efficient if set.
        Can be set to None to allow various input size time series.
        The default is None. 
    kernelInit : numpy array or LIST or INT, optional
        Initialisation of the kernel. If INT, random normal initialisation of size kernelInit.
        If array or LIST, then kernelInit is the kernel.
        The default is 8.
    kernTrainable : BOOL, optional
        Whether the kernels are trainable. Set to FALSE to compare to traditional wavelet decomposition. 
        The default is True.
    level : INT, optional
        Number of layers in the network.
        Ideally should be log2 of the time series length.
        If bigger, additional layers will be of size 1.
        The default is 1.
    lossCoeff : STRING, optional
        To specify which loss on the wavelet coefficient to compute.
        Can be None (no loss computed) or 'l1'' for the L1-norm of the coefficients.
        The default is 'l1'.
    kernelsConstraint : STRING, optional
        Specify which version of DeSpaWN to implement.
        Refers to the paper (https://arxiv.org/pdf/2105.00899.pdf) 
        [Section 4.4 Ablation Study] for more details.
        The default is 'CQF'.
    initHT : FLOAT, optional
        Value to initialise the Hard-thresholding coefficient.
        The default is 1.0.
    trainHT : BOOL, optional
        Whether the hard-thresholding coefficient is trainable or not.
        Set to FALSE to compare to traditiona wavelet decomposition.
        The default is True.

    Returns
    -------
    model1: a TF neural network with outputs the reconstructed signals and the loss on the wavelet coefficients
    model2: a TF neural network with outputs t the reconstructed signals and wavelet coefficients

    model1 and model2 share their architecture, weigths and parameters.
    Training one of the two changes both models
    """
    
    input_shape = (inputSize,1,1)
    inputSig = keras.layers.Input(shape=input_shape, name='input_Raw')
    g = inputSig
    if kernelsConstraint=='CQF':
        kern = impLay.Kernel(kernelInit, trainKern=kernTrainable)(g)
        kernelsG  = [kern for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    elif kernelsConstraint=='PerLayer':
        kernelsG  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    elif kernelsConstraint=='PerFilter':
        kernelsG  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsGT = kernelsG
        kernelsHT = kernelsH
    elif kernelsConstraint=='Free':
        kernelsG  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsGT = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsHT = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
    hl = []
    inSizel = []
    # Decomposition
    for lev in range(level):
        inSizel.append(tf.shape(g))
        hl.append(impLay.HardThresholdAssym(init=initHT,trainBias=trainHT)(impLay.HighPassWave()([g,kernelsH[lev]])))
        g  = impLay.LowPassWave()([g,kernelsG[lev]])
    g = impLay.HardThresholdAssym(init=initHT,trainBias=trainHT)(g)
    # save intermediate coefficients to output them
    gint = g
    # Reconstruction
    for lev in range(level-1,-1,-1):
        h = impLay.HighPassTrans()([hl[lev],kernelsHT[lev],inSizel[lev]])
        g = impLay.LowPassTrans()([g,kernelsGT[lev],inSizel[lev]])
        g = keras.layers.Add()([g,h])
    
    # Compute specified loss on coefficients
    if not lossCoeff:
        vLossCoeff = tf.zeros((1,1,1,1))
    elif lossCoeff=='l1':
        # L1-Sum
        vLossCoeff = tf.math.reduce_mean(tf.math.abs(tf.concat([gint]+hl,axis=1)),axis=1,keepdims=True)
    else:
        raise ValueError('Could not understand value in \'lossCoeff\'. It should be either \'l1\' or \'None\'')
    return keras.models.Model(inputSig,[g,vLossCoeff]), keras.models.Model(inputSig,[g,gint,hl[::-1]])
    ####  /!\ In tf > 2.2.0 each output variable is 1 output. The second model above output 3 variables and not level+2 as is tf 2.1.0
