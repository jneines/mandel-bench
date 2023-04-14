#!/usr/bin/env python3
import numpy as np


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    c = x + y[:, np.newaxis] * 1j
    c0 = c.copy()
    iterations = np.zeros_like(c, dtype=np.uint)

    for iteration in range(max_iterations):
        mask = np.abs(c) < 2
        c[mask] = c[mask] ** 2 + c0[mask]

        iterations[mask] += 1

    return iterations, {}
