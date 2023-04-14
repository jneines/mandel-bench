#!/usr/bin/env python3
from functools import partial

from dask.distributed import Client
import dask.array as da
import numpy as np


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    def mandel(c, max_iterations):
        c0 = c.copy()
        iterations = np.zeros_like(c, dtype=np.uint)
        for index in range(max_iterations):
            mask = np.abs(c) < 2.0
            c[mask] = c[mask] ** 2 + c0[mask]
            iterations[mask] += 1
        return iterations

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    c = x[:] + y[:, None] * 1j

    my_mandel = partial(mandel, max_iterations=max_iterations)

    client = Client("picm00.local:8786")

    futures = client.map(my_mandel, c)
    iterations = client.gather(futures)
    return iterations, {}
