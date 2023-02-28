#!/usr/bin/python3
# -*- coding: utf-8 -*-

from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
from numpy.typing import NDArray


def fetch_mnist(
    path: Path = Path("data/mnist"),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Do not modify. Downloads the MNIST dataset and caches in ``path``."""
    x, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, data_home=str(path)
    )
    return x, y


def convert_labels(y: NDArray[np.float64]) -> NDArray[np.int64]:
    r"""Convert 0-9 labels to 1/0 where 1 is even and 0 is odd."""
    # YOUR CODE GOES HERE >>>>>
    return y

    # <<<<<


def reshape_data(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    r"""Reshape each example into a row vector and each label to have shape (1, 1)."""
    # YOUR CODE GOES HERE >>>>>
    return x, y

    # <<<<<


def scale_examples(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Scale pixel features to be between 0 and 1."""
    # YOUR CODE GOES HERE >>>>>
    return x

    # <<<<<


def preprocess_data(
    path: Path = Path("data/mnist"),
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Do not modify. Download and return data, ready for training."""
    x, y = fetch_mnist(path)
    y = convert_labels(y)
    x, y = reshape_data(x, y)
    x = scale_examples(x)

    return x, y


def split_data(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    """Do not modify. Split the data into 80/20 training/testing split."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=25
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x, y = preprocess_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
