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
from mstorch.nn.basis import Linear


# Create random output of previous layer
prev_size = (1, np.random.randint(1, 11))
prev_output = np.random.uniform(0, 1, size=prev_size)

# Convert to tensor for pytorch
prev_output_t = torch.tensor(prev_output, requires_grad=False)

# Instantiate the linear basis object
size = np.random.randint(1, 11)
mlcv_linear = Linear(in_features=prev_size[1], out_features=size)
torch_linear = torch.nn.Linear(in_features=prev_size[1], out_features=size)

# Force torch to use our weights
torch_linear.weight = torch.nn.Parameter(torch.tensor(mlcv_linear.weight))
torch_linear.bias = torch.nn.Parameter(torch.tensor(mlcv_linear.bias))

# Do the computation
mlcv_forward = mlcv_linear.forward(input=prev_output)
torch_forward = torch_linear(prev_output_t).detach().numpy()

# Check element wise equality with some tolerance for float precision error
assert np.allclose(mlcv_forward, torch_forward)
