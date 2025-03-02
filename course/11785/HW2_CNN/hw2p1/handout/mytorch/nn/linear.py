import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False,weight_init_fn=None, bias_init_fn=None):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        if weight_init_fn is None:
            self.W = np.zeros((out_features,in_features))
        else:
            self.W =weight_init_fn(out_features,in_features)  # TODO
        if bias_init_fn is None:
            self.b = np.zeros(out_features)  # TODO
        else:
            self.b =bias_init_fn(out_features)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = np.array(A)  # TODO
        self.N = self.A.shape[0] # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = self.A@self.W.T + self.Ones@self.b.T  # TODO

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ@self.W # TODO
        self.dLdW = (dLdZ.T)@self.A  # TODO
        self.dLdb = (dLdZ.T)@self.Ones  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
