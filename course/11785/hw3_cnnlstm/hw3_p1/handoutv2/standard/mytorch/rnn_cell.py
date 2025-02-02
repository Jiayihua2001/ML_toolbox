import numpy as np
from mytorch.nn.activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).
        """
        # Compute the new hidden state
        h_t = self.activation(
            np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h_prev_t, self.W_hh.T) + self.b_hh
        )
        return h_t


    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).
        """
        batch_size = delta.shape[0]
        dz = self.activation.backward(state=h_t) * delta 
        # 1) Compute the averaged gradients 
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size 
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size 
        self.db_ih += np.mean(dz, axis=0) 
        self.db_hh += np.mean(dz, axis=0) 

        # 2) Compute dx, dh
        dx = np.dot(dz, self.W_ih) 
        dh = np.dot(dz, self.W_hh) 

        # 3) Return dx, dh
        return dx, dh
