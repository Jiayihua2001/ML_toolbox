import numpy as np
from layers import *

# This code is only for your reference for Sections 3.3 and 3.4


class MLP():
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            self.layers.append(Linear(in_size, out_size))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1]  # remove final ReLU

    def init_weights(self, weights):
        for i in range(len(weights)):
            self.layers[i * 2].W = weights[i].T

    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def train(mlp, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        # Forward pass
        y_pred = mlp.forward(X)
        
        # Compute loss
        loss = mse_loss(y_pred, y)
        
        # Backward pass
        dLdZ = mse_loss_grad(y_pred, y)
        mlp.backward(dLdZ)
        
        # Update weights
        for layer in mlp.layers:
            if isinstance(layer, Linear):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

if __name__ == '__main__':
    D = 24  # length of each feature vector
    layer_sizes = [8 * D, 8, 16, 4]
    mlp = MLP(layer_sizes)
    
    # Example data
    X = np.random.randn(100, 8 * D)  # 100 samples, each of size 8*D
    y = np.random.randn(100, 4)      # 100 target vectors, each of size 4
    
    # Train the MLP
    train(mlp, X, y, epochs=100, learning_rate=0.01)
