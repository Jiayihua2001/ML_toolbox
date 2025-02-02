import numpy as np
import scipy
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def __init__(self):
        pass
    def forward(self,Z):
        self.A  = 1/(1+np.exp(-Z))
        self.Z =Z
        return self.A
    def backward(self,dLdA):
        dAdZ = self.A -self.A**2
        dLdZ  = dLdA *dAdZ
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def __init__(self):
        pass
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    def backward(self,dLdA):
        dAdZ = 1-self.A**2
        dLdZ  = dLdA  * dAdZ
        return dLdZ

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def __init__(self):
        pass
    def forward(self,Z):
        self.A  = np.maximum(0,Z)
        self.Z = Z
        return self.A
    def backward(self,dLdA):
        dAdZ = (self.Z > 0).astype(float)
        dLdZ  = dLdA * dAdZ
        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def __init__(self):
        pass
    def forward(self,Z):
        self.A  = (Z/2)*(1+erf(Z/np.sqrt(2)))
        self.Z =Z
        return self.A
    def backward(self,dLdA):
        dAdZ = (1/2)*(1+erf(self.Z/np.sqrt(2)))+self.Z/np.sqrt(2*np.pi)*np.exp(-self.Z**2/2)
        dLdZ  = dLdA  * dAdZ
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        exp_z = np.exp(Z-np.max(Z,axis=1,keepdims=True)) # TODO
        self.A = exp_z/np.sum(exp_z,axis = 1,keepdims=True)


        return self.A

    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C = dLdA.shape

        # Initialize the final output dLdZ
        dLdZ = np.zeros_like(dLdA)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C)) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m != n:
                        J[m,n] = -self.A[i,m] * self.A[i,n]
                    else:
                        J[m,n] = self.A[i,m] * (1 - self.A[i,m])

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i] = dLdA[i]@J    # TODO

        return dLdZ