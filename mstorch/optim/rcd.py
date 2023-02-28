#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Iterable

import numpy as np
import numpy.typing as npt

from mstorch.optim.optimizer import Optimizer
from mstorch.nn.basis import Linear  # A small hint


class RCD(Optimizer):
    r"""Implements the Random Coordinate Descent optimization algorithm."""

    def __init__(self, layers: Iterable, learning_rate: float = 1e-3) -> None:
        r"""Initialize the RCD object.

        :param layers: Iterable of network components. E.g. (Linear object, ReLU object, etc.)
        :param learning_rate: gamma.
        """
        super().__init__(layers)  # Do not modify

        # We could also give this function to the students as a hint on how the library is supposed to work
        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def step(self, loss_grad: npt.NDArray[np.float64]) -> None:
        r"""Performs a single optimization step (parameter update)."""
        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<
