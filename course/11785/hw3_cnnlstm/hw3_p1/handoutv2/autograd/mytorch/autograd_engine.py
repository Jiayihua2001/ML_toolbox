import numpy as np
from typing import Optional, Union, List, Callable
from mytorch.utils import GradientBuffer


class Operation:
    def __init__(
        self,
        inputs: List[np.ndarray],
        output: np.ndarray,
        gradients_to_update: List[Optional[Union[np.ndarray, None]]],
        backward_operation: Callable,
    ):
        """
        Args:
            - inputs: operation inputs (List[np.ndarray])
            - outputs: operation output (Optional[Union[np.ndarray, List[np.ndarray]]])
            - gradients_to_update: parameter gradients if for parameter of ,
                        network or None (numpy.ndarray, None)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Note: You do not need to modify anything here
        """
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation

        self.i0_shp = self.inputs[0].shape
        self.i1_shp = None
        if len(self.inputs) > 1:
            self.i1_shp = self.inputs[1].shape
        self.bwd_op_name = self.backward_operation.__name__

    def __repr__(self):
        """
        Use this with print(operation) to help debug.
        """
        return f"Operation [{self.i0_shp}, {self.i1_shp}, {self.output.shape}, {self.gradients_to_update}, {self.bwd_op_name}]"


class Autograd:
    def __init__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        A check to make sure you don't create more than 1 Autograd at a time. You can remove
        this if you want to do multiple in parallel. We do not recommend this
        """
        if getattr(self.__class__, "_has_instance", False):
            raise RuntimeError("Cannot create more than 1 Autograd instance")
        self.__class__._has_instance = True

        self.gradient_buffer = GradientBuffer()
        self.operation_list = []

    def __del__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        Class destructor. We use this for testing purposes.
        """
        del self.gradient_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(
        self,
        inputs: List[np.ndarray],
        output: np.ndarray,
        gradients_to_update: List[Optional[Union[np.ndarray, None]]],
        backward_operation: Callable,
    ):
        """
        Adds operation to operation list and puts gradients in gradient buffer for tracking
        Args:
            - inputs: operation inputs (numpy.ndarray)
            - outputs: operation output (numpy.ndarray)
            - gradients_to_update: parameter gradients if for parameter of
                        network or None (numpy.ndarray, None)
                NOTE: Given the linear layer as shown in the writeup section
                    2.4 there are 2 kinds of inputs to an operation:
                    1) one that requires gradients to be internally tracked
                        ex. input (X) to a layer
                    2) one that requires gradient to be externally tracked
                        ex. weight matrix (W) of a layer (so we can track dW)
            - backward_operation: backward function for nn/functional.py.
                        When passing a function you don't need inputs or parentheses.
        Returns:
            No return required
        """
        if len(inputs) != len(gradients_to_update):
            raise Exception(
                "Number of inputs must match the number of gradients to update!"
            )

        # Add all of the inputs to the gradient buffer
        for input_tensor in inputs:
            self.gradient_buffer.add_spot(input_tensor)

        # Append an Operation object to the operation list
        operation = Operation(inputs, output, gradients_to_update, backward_operation)
        self.operation_list.append(operation)

    def backward(self, divergence):
        """
        Backpropagation through the operation list with a given divergence.
        This function should automatically update gradients of parameters by
        checking the gradients_to_update.
        
        Args:
            - divergence: loss value (float/double/int/long)
        Returns:
            No return required
        """
        # Initialize gradient for propagation based on divergence
        if np.isscalar(divergence):
            gradient_to_propagate = np.array(1.0)
        else:
            gradient_to_propagate = np.ones_like(divergence)

        # Traverse operations in reverse order for backpropagation
        for operation in reversed(self.operation_list):
            # Perform the backward operation, outputting gradients for each input
            gradients = operation.backward_operation(*operation.inputs, gradient_to_propagate)

            # Loop through inputs and corresponding gradients
            for i, (input_tensor, grad) in enumerate(zip(operation.inputs, gradients)):
                # Check if gradients need updating externally
                if operation.gradients_to_update[i] is not None:
                    # Reshape gradient to match expected shape, if necessary
                    expected_shape = operation.gradients_to_update[i].shape
                    if grad.shape != expected_shape:
                        grad = np.reshape(grad, expected_shape)

                    # Update gradient for external parameters
                    operation.gradients_to_update[i] += grad
                else:
                    # Update internally tracked gradients in the gradient buffer
                    self.gradient_buffer.update_param(input_tensor, grad)

            # Update gradient to propagate based on the first input's gradient
            gradient_to_propagate = gradients[0]

    def zero_grad(self):
        """
        Resets gradient buffer and operations list. No need to modify.
        """
        self.gradient_buffer.clear()
        self.operation_list = []




