#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from mstorch.nn.module import Module


class ReLU(Module):
    r"""Applies the ReLU function elementwise: :math:`max(0, x)`."""

    def __init__(self):
        super().__init__()  # Do not modify

        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def forward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Apply ReLU to each element in ``input``.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        :param input: A tensor of any dimension.
        :return: A tensor with ``input.shape`` shape and :math:`\text{ReLU}(x_ij)` elements.
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(input)

        # <<<<<

    def backward(self) -> npt.NDArray[np.float64]:
        r"""Calculate gradients of ReLU w.r.t :math:`x`.

        .. math::
            \nabla_{x}Y =
            \begin{cases}
            1, \text{ if } x > 0
            0, \text{ otherwise }
            \end{cases}

        :return: Gradient of ReLU activation with shape like ``input``
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(self.previous_input)  # A small hint here

        # <<<<<


class Sigmoid(Module):
    r"""Applies the Sigmoid function elementwise: :math:`\frac{1}{1+e^{-x}}`."""

    def __init__(self):
        super().__init__()  # Do not modify

        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def forward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Apply Sigmoid to each element in ``input``.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        :param input: A tensor of any dimension.
        :return: A tensor with ``input.shape`` shape and :math:`\text{Sigmoid}(x_ij)` elements.
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(input)

        # <<<<<

    def backward(self) -> npt.NDArray[np.float64]:
        r"""Calculate gradients of Sigmoid w.r.t :math:`x`.

        .. math::
            \nabla_{x}Y = y(1 - y)
            \text { Where y is the most recent output. }

        :return: Gradient of Sigmoid activation with shape like ``input``
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros_like(self.previous_input)  # A small hint here

        # <<<<<


if __name__ == "__main__":
    # Simulate output of basis
    size = (1, 5)  # 5 neurons in this layer
    basis_output = np.random.uniform(-1, 1, size=size)

    activation = ReLU()

    # Calculate element wise ReLU
    print(f"ReLU activation output:\n{activation.forward(basis_output)}\n")

    # Calculate gradient of ReLU
    print(f"ReLU grad w.r.t. x:\n{activation.backward()}\n\n")

    activation = Sigmoid()

    # Calculate element wise Sigmoid
    print(f"Sigmoid activation output:\n{activation.forward(basis_output)}\n")

    # Calculate gradient of Sigmoid
    print(f"Sigmoid grad w.r.t. x:\n{activation.backward()}")
