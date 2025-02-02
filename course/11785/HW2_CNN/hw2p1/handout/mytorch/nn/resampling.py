import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = (input_width-1) * self.upsampling_factor +1
        Z = np.zeros((batch_size, in_channels, output_width))  # Fixed
        for w in range(input_width):
            Z[:, :, w*self.upsampling_factor] = A[:, :, w]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width-1) // self.upsampling_factor +1 # Ensure integer division
        dLdA = np.zeros((batch_size, in_channels, input_width))  # Fixed
        for w in range(input_width):
            dLdA[:, :, w] = dLdZ[:, :, w *self.upsampling_factor]
        
        return dLdA



import numpy as np

class Downsample1d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        self.input_width =input_width
        output_width = (input_width + self.downsampling_factor - 1) // self.downsampling_factor  # Corrected output size calculation
        Z = A[:, :, ::self.downsampling_factor]  # Efficient slicing to downsample
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width =  self.input_width
        dLdA = np.zeros((batch_size, in_channels, input_width))

        # Distribute gradient to corresponding locations
        dLdA[:, :, ::self.downsampling_factor] = dLdZ
        return dLdA



class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = (input_height-1) * self.upsampling_factor +1
        output_width = (input_width-1) * self.upsampling_factor +1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))  

        for x in range(input_height):
            for y in range(input_width):
                Z[:, :, x * self.upsampling_factor, y * self.upsampling_factor] = A[:, :, x, y]  # Insert values

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = (output_height-1) // self.upsampling_factor +1  # Ensure integer division
        input_width = (output_width-1) // self.upsampling_factor +1  # Ensure integer division
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))  

        for x in range(input_height):
            for y in range(input_width):
                dLdA[:, :, x, y] = dLdZ[:, :, x * self.upsampling_factor, y * self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        
        batch_size, in_channels, input_height, input_width = A.shape
        self.input_height=input_height
        self.input_width =input_width
        output_height = (input_height-1) // self.downsampling_factor +1 # Ensure integer division
        output_width = (input_width-1) // self.downsampling_factor +1 # Ensure integer division
        Z = np.zeros((batch_size, in_channels, output_height, output_width))  

        for x in range(output_height):
            for y in range(output_width):
                Z[:, :, x, y] = A[:, :, min(input_height-1, x * self.downsampling_factor), min(input_width-1, y * self.downsampling_factor)]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = self.input_height # Ensure correct input height
        input_width = self.input_width # Ensure correct input width
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))  

        for x in range(output_height):
            for y in range(output_width):
                dLdA[:, :, min(input_height-1, x * self.downsampling_factor), min(input_width-1, y * self.downsampling_factor)] = dLdZ[:, :, x, y]

        return dLdA
