import numpy as np
from mytorch.autograd_engine import Autograd

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
"""

def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""

    return grad_output

def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""
    
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""
    
    a_grad = grad_output * np.ones(a.shape)
    b_grad = -grad_output * np.ones(b.shape)

    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""
    a_grad = np.dot(grad_output, b.T)
    b_grad = np.dot(a.T, grad_output)
    return a_grad, b_grad

def mul_backward(grad_output, a, b):
    """Backward for multiplication"""
    a_grad = grad_output * b
    b_grad = grad_output * a
    return a_grad, b_grad
    return NotImplementedError


def div_backward(grad_output, a, b):
    """Backward for division"""
    a_grad = grad_output / b
    b_grad = grad_output * (-a)/b**2
    return a_grad, b_grad
    return NotImplementedError


def log_backward(grad_output, a):
    """Backward for log"""
    a_grad = grad_output/a
    return a_grad 
    return NotImplementedError


def exp_backward(grad_output, a):
    """Backward of exponential"""
    a_grad = grad_output * np.exp(a)
    return a_grad
    # Removed redundant return NotImplementedError


def max_backward(grad_output, a):
    """Backward of max"""
    one_a = np.zeros_like(a)
    one_a[a > 0] = 1
    a_grad = grad_output * one_a
    return a_grad
    # Removed redundant return NotImplementedError


def sum_backward(grad_output, a):
    """Backward of sum"""
    a_grad = grad_output * np.ones_like(a)
    return a_grad
    # Implemented sum_backward
def tanh_backward(grad_output, a):
    a_grad = grad_output * (1-np.tanh(a)**2)
    return a_grad
def Sigmoid_backward(grad_output, a):
    sig = 1/(1+np.exp(-a))
    a_grad = grad_output * (sig-sig**2)

    return a_grad



def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """
    Backward pass for Softmax CrossEntropy Loss.
    """
    # Compute the gradient of the loss with respect to the logits (pred)
    # pred is already the softmax output, ground_truth is one-hot encoded
    batch_size = ground_truth.shape[0]
    exp_shifted = np.exp(pred)
    softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    grad_pred = (softmax - ground_truth) / batch_size   # Gradient scaled by batch size
    
    # No need to compute gradients for ground truth (return None)
    return grad_pred, None



