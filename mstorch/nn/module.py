#!/usr/bin/python3
# -*- coding: utf-8 -*-


class Module:
    r"""Base class for all neural network components; loosely based on PyTorch's nn.Module."""

    def __init__(self) -> None:
        """Do not modify."""
        self.freeze = True  # Layers with tunable parameters should set this to False
        return

    def get_parameters(self) -> tuple:
        r"""Do not modify. Implemented by layers that have tunable parameters."""
        return tuple()
