#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from mstorch.nn.module import Module


class L2Loss(Module):
    r"""Create a criterion that measures the L2 norm, error, between the input :math:`\hat{y} and target :math:`y`.

    .. math::
        z = \ell(y, \hat{y}) = \left\Vert y - \hat{y} \right_2 = \left[ \sum (y_i - \hat{y})^2 \right]^\frac1{2}
    """

    def __init__(self, epsilon: np.float64 = 10e-8) -> None:
        super().__init__()  # Do not modify

        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def forward(
        self, input: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Measure L2 norm of difference between input and target.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        .. math::
            z = \ell(y, \hat{y}) = \left\Vert y - \hat{y} \right_2 = \left[ \sum (y_i - \hat{y})^2 \right]^\frac1{2}

        :param input: :math:`\hat{y}` Output vector of :math`n` elements of the neural network.
        :param target: :math:`y` Ground truth label vector of :math`n` elements for training input data example.
        :return: L2 norm of :math`y - \hat{y}` vector.
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(input)  # A small hint here

        # <<<<<

    def backward(self) -> npt.NDArray[np.float64]:
        r"""Gradient of L2 norm with respect to :math:`\hat{y}.

        .. math::
            \nabla_\hat{y}z = \frac{\partial z}{\partial \hat{y}_i} = \frac{-1}{z} (y - \hat{y})

        .. note::
            To avoid :math:`\frac0{0}` or division by :math`0`, add a small :math:`\epsilon`.

        :param input: :math:`\hat{y}` Output vector of :math`n` elements of the neural network.
        :param target: :math:`y` Ground truth label vector of :math`n` elements for training input data example.
        :return: Gradient of L2 norm with shape like ``input``, :math:`\hat{y}`.
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(self.previous_input)  # A small hint here

        # <<<<<


if __name__ == "__main__":
    loss = L2Loss()

    # Simulate network output and label target
    size = (5, 1)  # Our "network" has 5 output neurons
    network_output = np.random.uniform(-10, 10, size=size)  # y_hat
    target = np.random.uniform(-10, 10, size=size)  # y

    # Calculate L2 loss between network output and target
    print(f"L2 loss:\n{loss.forward(input=network_output, target=target)}\n")

    # Calculate gradient of loss w.r.t. network output (y_hat)
    print(f"L2 grad w.r.t. network output:\n{loss.backward()}")
