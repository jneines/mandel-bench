#!/usr/bin/env python
import concurrent.futures as cf
from functools import partial

import numpy as np


def mandel_row(y_value, x_min, x_max, max_iterations, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.array([y_value])

    c = x + y[:, None] * 1j
    c0 = c.copy()
    iterations = np.zeros_like(c, dtype=np.uint)

    for iteration in range(max_iterations):
        mask = np.abs(c) < 2.0
        c[mask] = c[mask] ** 2 + c0[mask]
        iterations[mask] += 1
    return iterations[0, :]


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):

    my_mandel_row = partial(
        mandel_row,
        x_min=x_min,
        x_max=x_max,
        max_iterations=max_iterations,
        resolution=resolution,
    )

    y = np.linspace(y_min, y_max, resolution)

    with cf.ProcessPoolExecutor() as executor:
        iterations = executor.map(my_mandel_row, y)
    iterations = list(iterations)
    return iterations, {}
