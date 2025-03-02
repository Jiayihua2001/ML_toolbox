# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys
import pdb

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # <---------------------

        self.conv1 = Conv1d(24,8,8,4)
        self.conv2 = Conv1d(8,16,1,1)
        self.conv3 = Conv1d(16,4,1,1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, Flatten()] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        print("w1 shape:", w1.shape)
        print("w2 shape:", w2.shape)
        print("w3 shape:", w3.shape)
        self.conv1.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w1), (8, 8, 24)), (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w2), (16, 1, 8)), (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w3), (4, 1, 16)), (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for i, layer in enumerate(self.layers):
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ  # Initialize dLdA
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24,2,2,2)
        self.conv2 = Conv1d(2,8,2,2)
        self.conv3 = Conv1d(8,4,2,1)
        self.layers = [ self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, Flatten()] # TODO: Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        print("w1 shape:", w1.shape)
        print("w2 shape:", w2.shape)
        print("w3 shape:", w3.shape)
        self.conv1.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w1[:, :2]), (2, 8, 24))[:, :2, :], (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w2[:, :8]), (8, 4, 2))[:, :2, :], (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(np.reshape(np.transpose(w3), (4, 2, 8)), (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for i, layer in enumerate(self.layers):
            Z = layer.forward(Z)
            print(f"Layer {i} output shape: {Z.shape}")
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
