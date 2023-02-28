#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Iterable

import numpy as np
import numpy.typing as npt

from mstorch.optim.optimizer import Optimizer
from mstorch.nn.basis import Linear  # A small hint


class Adam(Optimizer):
    r"""Implements the Adam optimization algorithm."""

    def __init__(
        self,
        layers: Iterable,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        r"""Initialize the Adam object.

        :param layers: Iterable of network components. E.g. (Linear object, ReLU object, etc.)
        :param learning_rate: gamma.
        :param betas: (Beta1, Beta2): Coefficients used for computing running averages of gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        :param weight_decay: Weight decay (L2 penalty); 0 -> previous weights are not incorporated.
        """
        super().__init__(layers)  # Do not modify

        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<

    def step(self, loss_grad: npt.NDArray[np.float64]) -> None:
        r"""Performs a single optimization step (parameter update)."""
        # TODO: Replace below with your code >>>>>
        pass

        # <<<<<
