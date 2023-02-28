#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numpy import float64
from numpy.typing import NDArray


class Optimizer:
    r"""Base class for all optimizers; loosely based on PyTorch's optim.Optimizer."""

    def __init__(self, layers) -> None:
        r"""Do not modify."""
        self.layers = layers

    def zero_grad(self) -> None:
        r"""Resets gradients of all optimized parameters."""
        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def step(self, loss_grad: NDArray[float64]) -> None:
        r"""Do not modify. Performs a single optimization step (parameter update)."""
        raise NotImplementedError("step method not implemented.")
