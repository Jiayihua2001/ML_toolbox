import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        self.r = None
        self.z = None
        self.n = None
        self.x = None
        self.hidden = None

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        x = np.expand_dims(x, 1)  
        h = np.expand_dims(h, 1) 

        self.r = self.r_act(self.Wrx @ x + self.brx.reshape((-1, 1)) + self.Wrh @ h + self.brh.reshape(-1, 1))
        self.z = self.z_act(self.Wzx @ x + self.bzx.reshape((-1, 1)) + self.Wzh @ h + self.bzh.reshape((-1, 1)))
        self.n = self.h_act(self.Wnx @ x + self.bnx.reshape((-1, 1)) + self.r * (self.Wnh @ h + self.bnh.reshape(-1, 1)))
        h_t = (1 - self.z) * self.n + self.z * h
        #squeeze r z n
        self.r = np.squeeze(self.r)
        self.z = np.squeeze(self.z)
        self.n = np.squeeze(self.n)
        # check the dimensions
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        
        h_t = np.squeeze(h_t)
        assert h_t.shape == (self.h,)  
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        backward wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of backward wrt loss from next layer at
                the same time-step and backward wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            backward of the loss wrt the input x.

        dh: (1, hidden_dim)
            backward of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly


        delta = np.reshape(delta, (-1, 1))
        # delta (h,1)

        r = np.reshape(self.r, (-1, 1))
        z = np.reshape(self.z, (-1, 1))
        n = np.reshape(self.n, (-1, 1))
        h_prev = np.reshape(self.hidden, (-1, 1))
        x = np.reshape(self.x, (-1, 1)).T  # (1,d)

        dn = delta * (1 - z)
        dz = delta * (-n + h_prev)

        d_n_affine = dn * self.h_act.backward(n)  # (n,1)
        self.dWnx = d_n_affine @ x
        self.dbnx = np.squeeze(d_n_affine)
        dr = d_n_affine * (self.Wnh @ (np.expand_dims(self.hidden, 1)) + self.bnh.reshape(-1, 1))
        self.dWnh = d_n_affine * r @ h_prev.T
        self.dbnh = np.squeeze(d_n_affine * r)

        dz_affine = dz * self.z_act.backward()
        self.dWzx = dz_affine @ x
        self.dbzx = np.squeeze(dz_affine)
        self.dWzh = dz_affine @ h_prev.T
        self.dbzh = np.squeeze(dz_affine)

        dr_affine = dr * self.r_act.backward()
        self.dWrx = dr_affine @ x
        self.dbrx = np.squeeze(dr_affine)
        self.dWrh = dr_affine @ h_prev.T
        self.dbrh = np.squeeze(dr_affine)

        dx = np.zeros((1, self.d))
        dx += d_n_affine.T @ self.Wnx
        dx += dz_affine.T @ self.Wzx
        dx += dr_affine.T @ self.Wrx
        dx = np.squeeze(dx)

        dh_prev = np.zeros((1, self.h))
        dh_prev += (delta * z).T
        dh_prev += (d_n_affine * r).T @ self.Wnh
        dh_prev += dz_affine.T @ self.Wzh
        dh_prev += dr_affine.T @ self.Wrh
        dh_prev = np.squeeze(dh_prev)
        # check the dimensions
        assert dx.shape == (self.d,)
        assert dh_prev.shape == (self.h,)

        return dx, dh_prev