#!/usr/bin/env python
import multiprocessing as mp
from functools import partial

import numpy as np


def mandel_row(y, x_min, x_max, max_iterations, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.array([y])

    c = x[:] + y[:, np.newaxis] * 1j
    c0 = c.copy()
    row = np.zeros_like(c, dtype=np.uint)

    for iteration in range(max_iterations):
        mask = np.abs(c) < 2.0
        c[mask] = c[mask] ** 2 + c0[mask]
        row[mask] += 1

    return row[0, :]


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):

    my_mandel_row = partial(
        mandel_row,
        x_min=x_min,
        x_max=x_max,
        max_iterations=max_iterations,
        resolution=resolution,
    )

    y = np.linspace(y_min, y_max, resolution)

    with mp.Pool() as pool:
        iterations = pool.map(my_mandel_row, y)
    return iterations, {}
