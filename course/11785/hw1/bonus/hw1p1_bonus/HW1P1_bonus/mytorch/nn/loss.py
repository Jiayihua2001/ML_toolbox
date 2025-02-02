import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = self.Y.shape[0]  # TODO
        self.C = self.Y.shape[1]  # TODO
        se = (self.A -self.Y)**2
        sse = np.sum(se)  # TODO
        mse = sse/(self.N*self.C)

        return mse

    def backward(self):

        dLdA = 2*(self.A -self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0]   # TODO
        C = self.A.shape[1]   # TODO

        Ones_C = np.ones(C)  # TODO
        Ones_N = np.ones(N)  # TODO
        
        exp_A = np.exp(A -np.max(A,axis=1,keepdims=True))
        self.softmax = exp_A/np.sum(exp_A,axis=1,keepdims=True)  # TODO
        crossentropy = (-Y*np.log(self.softmax))@Ones_C  # TODO
        sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        L = sum_crossentropy / N
        self.N = N
        return L

    def backward(self):

        dLdA = (self.softmax - self.Y)/self.N # TODO

        return dLdA
