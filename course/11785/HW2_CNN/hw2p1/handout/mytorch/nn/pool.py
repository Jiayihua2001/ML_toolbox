import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_width, kernel_height = self.kernel,self.kernel

        output_width = input_width - kernel_width + 1
        output_height = input_height - kernel_height + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.max(
                    A[:, :, i:i+kernel_width, j:j+kernel_height], axis=(2, 3)
                )
        self.A = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        kernel_width, kernel_height = self.kernel, self.kernel
        input_width = output_width + kernel_width - 1
        input_height = output_height + kernel_height - 1

        dLdA = np.zeros_like(self.A)

        for i in range(output_width):
            for j in range(output_height):
                A_slice = self.A[:, :, i:i+kernel_width, j:j+kernel_height]
                max_A = np.max(A_slice, axis=(2, 3), keepdims=True)
                mask = (A_slice == max_A)
                dLdA[:, :, i:i+kernel_width, j:j+kernel_height] += mask * dLdZ[:, :, i, j][:, :, None, None]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_width, kernel_height = self.kernel,self.kernel

        output_width = input_width - kernel_width + 1
        output_height = input_height - kernel_height + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.mean(
                    A[:, :, i:i+kernel_width, j:j+kernel_height], axis=(2, 3)
                )
        self.A = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        kernel_width, kernel_height = self.kernel,self.kernel
        input_width = output_width + kernel_width - 1
        input_height = output_height + kernel_height - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:i+kernel_width, j:j+kernel_height] += dLdZ[:, :, i, j][:,:,None,None] / (kernel_width * kernel_height)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)  # Implement downsampling logic

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.maxpool2d_stride1.forward(A)
      
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)  # Implement downsampling logic

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_stride1)

        return dLdA
