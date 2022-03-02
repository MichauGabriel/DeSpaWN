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
import tensorflow as tf

class Kernel(tf.keras.layers.Layer):
    def __init__(self, kernelInit=8, trainKern=True, **kwargs):
        self.trainKern  = trainKern
        if isinstance(kernelInit,int):
            self.kernelSize = kernelInit
            self.kernelInit = 'random_normal'
        else:
            self.kernelSize = kernelInit.__len__()
            self.kernelInit = tf.constant_initializer(kernelInit)
        super(Kernel, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(shape       = (self.kernelSize,1,1,1),
                                      initializer = self.kernelInit,
                                      trainable   = self.trainKern, name='kernel')
        super(Kernel, self).build(input_shape)
    def call(self, inputs):
        return self.kernel

class LowPassWave(tf.keras.layers.Layer):
    """
    Layer that performs a convolution between its two inputs with stride (2,1)
    """
    def __init__(self, **kwargs):
        super(LowPassWave, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LowPassWave, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d(inputs[0], inputs[1], padding="SAME", strides=(2, 1))

class HighPassWave(tf.keras.layers.Layer):
    """
    Layer that performs a convolution between its two inputs with stride (2,1).
    Performs first the reverse alternative flip on the second inputs
    """
    def __init__(self, **kwargs):
        super(HighPassWave, self).__init__(**kwargs)

    def build(self, input_shape):
        self.qmfFlip = tf.reshape(tf.Variable([(-1)**(i) for i in range(input_shape[1][0])],
                                              dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HighPassWave, self).build(input_shape)

    def call(self, inputs):
        # print(self.qmfFlip)
        return tf.nn.conv2d(inputs[0], tf.math.multiply(tf.reverse(inputs[1],[0]),self.qmfFlip),
                            padding="SAME", strides=(2, 1))

class LowPassTrans(tf.keras.layers.Layer):
    """
    Layer that performs a convolution transpose between its two inputs with stride (2,1).
    The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
    """
    def __init__(self, **kwargs):
        super(LowPassTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LowPassTrans, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d_transpose(inputs[0], inputs[1], inputs[2], padding="SAME", strides=(2, 1))

class HighPassTrans(tf.keras.layers.Layer):
    """
    Layer that performs a convolution transpose between its two inputs with stride (2,1).
    Performs first the reverse alternative flip on the second inputs
    The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
    """
    def __init__(self, **kwargs):
        super(HighPassTrans, self).__init__(**kwargs)

    def build(self, input_shape):
        self.qmfFlip     = tf.reshape(tf.Variable([(-1)**(i) for i in range(input_shape[1][0])],
                                                  dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HighPassTrans, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.conv2d_transpose(inputs[0], tf.math.multiply(tf.reverse(inputs[1],[0]),self.qmfFlip),
                                      inputs[2], padding="SAME", strides=(2, 1))

class HardThresholdAssym(tf.keras.layers.Layer):
    """
    Learnable Hard-thresholding layers
    """
    def __init__(self, init=None, trainBias=True, **kwargs):
        if isinstance(init,float) or isinstance(init,int):
            self.init = tf.constant_initializer(init)
        else:
            self.init = 'ones'
        self.trainBias = trainBias
        super(HardThresholdAssym, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thrP = self.add_weight(shape     = (1,1,1,1), initializer=self.init,
                                    trainable = self.trainBias,      name='threshold+')
        self.thrN = self.add_weight(shape     = (1,1,1,1), initializer=self.init,
                                    trainable = self.trainBias,      name='threshold-')

        super(HardThresholdAssym, self).build(input_shape)

    def call(self, inputs):
        return tf.math.multiply(inputs,tf.math.sigmoid(10*(inputs-self.thrP))+\
                                       tf.math.sigmoid(-10*(inputs+self.thrN)))
