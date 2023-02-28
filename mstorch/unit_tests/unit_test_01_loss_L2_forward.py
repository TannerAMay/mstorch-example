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
from mstorch.nn.loss import L2Loss


# Create random input and target data
size = (np.random.randint(1, 11), 1)
input = np.random.uniform(-10, 10, size=size)
target = np.random.uniform(-10, 10, size=size)

# Convert to tensors so pytorch can use them
input_t = torch.tensor(input, requires_grad=False)
target_t = torch.tensor(target, requires_grad=False)

# Instantiate the loss objects
mlcv_l2 = L2Loss()
torch_l2 = torch.nn.MSELoss(reduction="sum")

# Do the computation
mlcv_forward = mlcv_l2.forward(input, target)
torch_forward = torch.sqrt(torch_l2(input_t, target_t)).detach().numpy().item()

# Check element wise equality with some tolerance for float precision error
assert np.allclose(mlcv_forward, torch_forward)
