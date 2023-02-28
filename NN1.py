#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from mstorch.nn.module import Module
from mstorch.nn.basis import Linear
from mstorch.nn.activation import Sigmoid
from mstorch.nn.loss import L2Loss

# Choose your favorite optimizer
from mstorch.optim.adam import Adam
from mstorch.optim.rcd import RCD


class NN1(Module):
    def __init__(self) -> None:
        super().__init__()

        # Create list of layers in order of propagation
        self.layers = [Linear(28 * 28, 1), Sigmoid()]  # Output activation is optional

    def forward(self, input: npt.NDArray) -> npt.NDArray[np.float64]:
        r"""Forward propagate input.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        :param input: A row vector with the same number of features as first layer is expecting.
        :return: Output of last layer in network.
        """
        logits = self.layers[0].forward(input)
        return logits

    def backward(self) -> None:
        r"""Calculate the gradient of each layer in the model."""
        self.layers[0].backward()
        return


if __name__ == "__main__":
    # Create fake MNIST image as a ROW vector
    train_example = np.random.uniform(size=(1, 28 * 28))

    train_label = np.ones((1, 1))  # Create fake even label

    model = NN1()
    loss_fn = L2Loss()
    optimizer = Adam(model.layers)  # To try RCD, simply change `Adam` to `RCD`

    # Reset gradients
    optimizer.zero_grad()

    # Feed forward a single training example
    model_output = model.forward(train_example)

    # Compute the error between the prediction and the label
    loss_initial = loss_fn.forward(model_output, train_label)
    print(f"Initial loss: {loss_initial}")

    # Compute the gradients for each layer in the model and the loss function
    loss_grad = loss_fn.backward()
    model.backward()

    # Backpropagation
    optimizer.step(loss_grad)

    # Feed forward the same example to see if the model improved
    model_output = model.forward(train_example)

    loss_optim = loss_fn.forward(model_output, train_label)
    print(f"Loss after one optimization step: {loss_optim}")
    print(
        f"Loss after optimization should be lower than before, is it? {loss_optim < loss_initial}"
    )
