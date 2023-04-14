#!/usr/bin/env python3
import numpy as np
from numba import vectorize, uint32, uint64, float32, float64, complex64, complex128


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    @vectorize(
        [
            uint32(complex64, uint32),
            uint64(complex128, uint64),
        ]
    )
    def mandel(c, max_iterations):
        c0 = c
        for iteration in range(max_iterations):
            c = c ** 2 + c0
            if abs(c) > 2.0:
                break
        return iteration

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    c = x[:] + y[:, None] * 1j

    iterations = mandel(c, max_iterations)
    return iterations, {}
