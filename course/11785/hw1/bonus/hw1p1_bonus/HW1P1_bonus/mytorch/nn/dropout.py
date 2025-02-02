# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # Generate mask and apply to x
            self.mask = np.random.binomial(1, 1-self.p, size=x.shape)
            print(f"Mask: {self.mask}")  # Debug: Print the mask
            x = x * self.mask
            print(f"x after applying mask: {x}")  # Debug: Print x after mask
            # Scale x
            x = x / (1-self.p)
            print(f"x after scaling: {x}")  # Debug: Print x after scaling
            return x

        else:
            return x

    def backward(self, delta):
        # Multiply mask with delta and return
        print(f"Delta: {delta}")  # Debug: Print delta
        result = delta * self.mask
        print(f"Result after applying mask: {result}")  # Debug: Print result
        return result