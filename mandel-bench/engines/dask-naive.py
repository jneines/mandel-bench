#!/usr/bin/env python3
from functools import partial

from dask.distributed import Client


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    def mandel_row(y, x_min, x_max, max_iterations, resolution):
        xs = [
            x_min + (x_max - x_min) / (resolution - 1) * index
            for index in range(resolution)
        ]
        row = []
        for x in xs:
            c = complex(x, y)
            c0 = c
            for iteration in range(max_iterations):
                c = c ** 2 + c0
                if abs(c) > 2.0:
                    break
            row.append(iteration)

        return row

    y = [
        y_min + (y_max - y_min) / (resolution - 1) * index
        for index in range(resolution)
    ]

    iterations = []

    my_mandel_row = partial(
        mandel_row,
        x_min=x_min,
        x_max=x_max,
        max_iterations=max_iterations,
        resolution=resolution,
    )

    client = Client("picm00.local:8786")

    futures = client.map(my_mandel_row, y)
    iterations = client.gather(futures)

    return iterations, {}
