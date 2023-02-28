#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Import init
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third party libraries for ground truth testing
import torch
import numpy as np

# Your code tests start here:
# To debug in pudb3
# Highlight the line of code below below
# Type 't' to jump 'to' it
# Type 's' to 'step' deeper
# Type 'n' to 'next' over
# Type 'f' or 'r' to finish/return a function call and go back to caller
from mstorch.nn.activation import ReLU


# Create random output of basis
layer_size = (1, np.random.randint(1, 11))
basis_output = np.random.uniform(0, 1, size=layer_size)

# Convert to tensor for pytorch
basis_output_t = torch.tensor(basis_output, requires_grad=True)

# Instantiate the ReLU activation object
mlcv_relu = ReLU()
torch_relu = torch.nn.ReLU()

# Do the forward computation
mlcv_relu.forward(input=basis_output)
torch_forward = torch_relu(basis_output_t)
torch_forward.retain_grad()

mlcv_backward = mlcv_relu.backward()

raise NotImplementedError("Test not complete below this point.")
# No time to finish this test
# TODO: Complete this test

torch_forward.backward()
torch_backward = basis_output_t.detach().numpy()

# Check element wise equality with some tolerance for float precision error
assert np.allclose(mlcv_backward, torch_backward)
