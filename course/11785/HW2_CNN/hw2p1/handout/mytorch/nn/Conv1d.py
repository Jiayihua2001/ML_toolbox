# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))
        
        for x in range(output_size):
            Z[:, :, x] = np.tensordot(A[:, :, x:x+self.kernel_size], self.W, axes=([1, 2], [1, 2])) + self.b
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape

        dLdA = np.zeros_like(self.A)
        for x in range(output_size):
            self.dLdW += np.tensordot(dLdZ[:, :, x], self.A[:, :, x:x+self.kernel_size], axes=([0], [0]))
            self.dLdb += np.sum(dLdZ[:, :, x], axis=0)
            dLdA[:, :, x:x+self.kernel_size] += np.tensordot(dLdZ[:, :, x], self.W, axes=([1], [0]))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d_stride1 and Downsample1d instances
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)
        Z_conv = self.conv1d_stride1.forward(A_padded)
        Z = self.downsample1d.forward(Z_conv)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        dLdZ_upsampled = self.downsample1d.backward(dLdZ)
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_upsampled)

        # Unpad the gradient
        dLdA = dLdA_padded[:, :, self.pad:-self.pad] if self.pad != 0 else dLdA_padded

        return dLdA
