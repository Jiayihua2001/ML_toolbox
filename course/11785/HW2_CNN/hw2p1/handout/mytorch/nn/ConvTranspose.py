import numpy as np
from resampling import Upsample1d, Upsample2d
from Conv1d import Conv1d
from Conv2d import Conv2d


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d instance
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d(in_channels, out_channels, kernel_size, stride=1,
                                     weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
   
        A_upsampled = self.upsample1d.forward(A)
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        delta_out = self.conv1d_stride1.backward(dLdZ)
        dLdA = self.upsample1d.backward(delta_out)

        return dLdA


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor
        self.upsample2d = Upsample2d(upsampling_factor)
        self.conv2d_stride1 = Conv2d(in_channels, out_channels, kernel_size, stride=1,
                                     weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
 
        A_upsampled = self.upsample2d.forward(A)
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        delta_out = self.conv2d_stride1.backward(dLdZ)
        dLdA = self.upsample2d.backward(delta_out)

        return dLdA
