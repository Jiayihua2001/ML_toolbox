import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, _, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        Z[b, c_out, h, w] = np.sum(
                            A[b, :, h:h+self.kernel_size, w:w+self.kernel_size] * self.W[c_out, :, :, :]
                        ) + self.b[c_out]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, _, input_height, input_width = self.A.shape
        _, _, output_height, output_width = dLdZ.shape
        dLdA = np.zeros_like(self.A)

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        self.dLdW[c_out] += dLdZ[b, c_out, h, w] * self.A[b, :, h:h+self.kernel_size, w:w+self.kernel_size]
                        self.dLdb[c_out] += dLdZ[b, c_out, h, w]
                        dLdA[b, :, h:h+self.kernel_size, w:w+self.kernel_size] += dLdZ[b, c_out, h, w] * self.W[c_out]

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d_stride1 and Downsample2d instances
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)  # Assuming Downsample2d is defined elsewhere

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

        # Call Conv2d_stride1
        Z_conv = self.conv2d_stride1.forward(A_padded)

        # Downsample
        Z = self.downsample2d.forward(Z_conv)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_upsampled)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dLdA_padded

        return dLdA
