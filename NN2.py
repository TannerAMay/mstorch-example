import numpy as np
import numpy.typing as npt

from mstorch.nn.module import Module
from mstorch.optim.optimizer import Optimizer
from mstorch.nn.basis import Linear
from mstorch.nn.activation import ReLU, Sigmoid
from mstorch.nn.loss import L2Loss

# Choose your favorite optimizer
from mstorch.optim.adam import Adam
from mstorch.optim.rcd import RCD

from data_processing import preprocess_data, split_data


class NN2(Module):
    def __init__(self) -> None:
        super().__init__()  # Do not modify

        # YOUR CODE GOES HERE >>>>>

        # <<<<<

    def forward(self, input: npt.NDArray) -> npt.NDArray[np.float64]:
        r"""Forward propagate input.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        :param input: A row vector with the same number of features as first layer is expecting.
        :return: Output of last layer in network.
        """
        # YOUR CODE GOES HERE >>>>>
        return input

        # <<<<<

    def backward(self) -> None:
        r"""Calculate the gradient of each layer in the model."""
        # YOUR CODE GOES HERE >>>>>
        return

        # <<<<<


def train(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    model: NN2,
    optimizer: Optimizer,
    loss_fn: L2Loss,
) -> tuple[float, int]:
    r"""Train the model for one epoch through the training data.

    :param x: 3-dimensional numpy array of training examples.
    :param y: 3-dimensional numpy array of training labels.
    :param model: The model to train.
    :param optimizer: Optimizer to use to update model parameters.
    :param loss_fn: Loss to use to calculate error between model prediction and ground truth (y).
    :return: (total loss, total correct). Where correct is defined as output greater than 0.5 is a 1, otherwise 0.
    """
    # YOUR CODE GOES HERE >>>>>
    return np.inf, 0

    # <<<<<


def test(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.int64], model: NN2
) -> tuple[float, int]:
    r"""Test the model on unseen data.

    :param x: 3-dimensional numpy array of training examples.
    :param y: 3-dimensional numpy array of training labels.
    :param model: The model to train.
    :return: (total loss, total correct). Where correct is defined as output greater than 0.5 is a 1, otherwise 0.
    """
    # YOUR CODE GOES HERE >>>>>
    return np.inf, 0

    # <<<<<


if __name__ == "__main__":
    r"""Get the data using data_processing.py functions; create the model, loss, and optimizer objects, train the model
    to at least 75% accuracy, test the model on unseen data to ensure it is not over fitted (ie still >75% accurate).
    """
    # YOUR CODE GOES HERE >>>>>

    # <<<<<
