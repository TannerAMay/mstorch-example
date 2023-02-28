#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from mstorch.nn.module import Module


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xW^T + b`"""

    def __init__(self, in_features: int, out_features: int) -> None:
        r"""Initialize Linear transformation and its parameters with HE initialization.

        .. note:
            The parameter matrices can be of shape either (I, O) or (O, I). Just be consistent!
            If you need help deciding, know that PyTorch uses (O, I), and is easier!

        .. note:
            If you want to parallel PyTorch as much as possible:
            bias should have shape (out_features,) NOT (out_features, 1)!!!

        Recall HE Initialization:
        Sample :math:`W_k` from a Normal distribution such that:
        .. math:
            W_k ~ N(0, \frac2{|W_k-1|})
            Where :math:`|W_k|` is the out_features, the number of neurons, in the :math:`k^th` layer.

        :param in_features: Size of each input sample; i.e. the number of neurons in the previous layer.
        :param out_features: Size of each output sample; i.e. the number of neurons in this layer.
        """
        super().__init__()  # Do not modify

        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def get_parameters(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Return member variables containing tunable parameters."""
        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def forward(self, input: npt.NDArray) -> npt.NDArray[np.float64]:
        r"""Apply linear transform to ``input`` vector.

        .. note::
            When using real PyTorch, you would not call this function. But since we are not doing computational graphs,
            autodiff, or autograd, we will just call this manually.

        :param input: A row vector with ``self.in_features number of entries.
        :return: A row vector with ``self.out_features`` number of entries.
        """
        # TODO: Replace below with your code >>>>>
        return np.zeros((1, self.out_features))

        # <<<<<

    def backward(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate gradients of linear transform w.r.t. :math:`W`, w.r.t :math:`x`, w.r.t. :math:`bias`.

        .. math::
            \nabla_{W}Y = x
            \nabla_{x}Y = W
            \nabla_{bias}Y = 1

        :return: Tuple of gradients (w.r.t. :math:`W`, w.r.t. :math:`x`, w.r.t. :math:`bias`).
        """
        # TODO: Replace below with your code >>>>>
        # A small hint here
        return np.zeros_like(self.previous_input), np.zeros_like(self.weight), np.zeros_like(self.bias)

        # <<<<<


if __name__ == "__main__":
    INPUT_SIZE = 3

    linear = Linear(in_features=INPUT_SIZE, out_features=5)

    # Simulate previous layer output
    prev_output = np.random.uniform(0, 1, (1, INPUT_SIZE))

    # Feel free to modify prev_output and manually change linear.weight and linear.bias for testing

    # Calculate linear basis using HE Initialized weights
    print(f"Linear basis output:\n{linear.forward(prev_output)}\n")

    # Calculate gradients of linear basis
    grad_lin_wrt_w, grad_lin_wrt_x, grad_lin_wrt_b = linear.backward()
    print(f"Linear basis grad w.r.t. W:\n{grad_lin_wrt_w}\n")
    print(f"Linear basis grad w.r.t. x:\n{grad_lin_wrt_x}\n")
    print(f"Linear basis grad w.r.t. x:\n{grad_lin_wrt_b}")
