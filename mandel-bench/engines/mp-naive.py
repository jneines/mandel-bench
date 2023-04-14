#!/usr/bin/env python
import multiprocessing as mp
from functools import partial

# from .naive import calculate as naive_calculate


def mandel_row(y, x_min, x_max, max_iterations, resolution):
    x = [
        x_min + (x_max - x_min) / (resolution - 1) * index
        for index in range(resolution)
    ]
    row = []
    for _x in x:
        c = complex(_x, y)
        c0 = c
        for iteration in range(max_iterations):
            c = c ** 2 + c0
            if abs(c) > 2.0:
                break
        row.append(iteration)
    return row


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):

    my_mandel_row = partial(
        mandel_row,
        x_min=x_min,
        x_max=x_max,
        max_iterations=max_iterations,
        resolution=resolution,
    )

    y = [
        y_min + (y_max - y_min) / (resolution - 1) * index
        for index in range(resolution)
    ]

    with mp.Pool() as pool:
        iterations = pool.map(my_mandel_row, y)
    return iterations, {}
