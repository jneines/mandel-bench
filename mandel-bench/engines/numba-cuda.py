#!/usr/bin/env python3
import math

import numpy as np
from numba import cuda


@cuda.jit
def mandel(_c, iterations, max_iterations):
    x, y = cuda.grid(2)
    if x < iterations.shape[0] and y < iterations.shape[1]:
        c = _c[x, y]
        c0 = c
        for iteration in range(max_iterations):
            c = c ** 2 + c0
            if abs(c) > 2.0:
                break
        iterations[x, y] = iteration


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    c = x + y[:, None] * 1j
    iterations = np.zeros_like(c, dtype=np.uint)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(iterations.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(iterations.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mandel[blockspergrid, threadsperblock](c, iterations, max_iterations)
    return iterations, {}
