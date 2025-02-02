# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

class Sigmoid(Activation):
    """
    Sigmoid activation function
    """
    def forward(self, Z):

        self.A = Z
        self.npVal = np.exp(-self.A)
        return 1 / (1 + self.npVal)

    def backward(self):

        dAdZ = self.npVal / (1 + self.npVal) ** 2
        return dAdZ

class Tanh(Activation):
    """
    Modified Tanh to work with BPTT.
    The tanh(x) result has to be stored elsewhere otherwise we will
    have to store results for multiple timesteps in this class for each cell,
    which could be considered bad design.

    Now in the derivative case, we can pass in the stored hidden state and
    compute the derivative for that state instead of the "current" stored state
    which could be anything.
    """
    def forward(self, Z):

        self.A = Z
        self.tanhVal =  np.tanh(self.A)
        return self.tanhVal

    def backward(self, state=None):
        if state is not None:
            dAdZ = 1 - state**2
            return dAdZ 
        else:
            dAdZ = 1 - self.tanhVal * self.tanhVal
            return dAdZ 
