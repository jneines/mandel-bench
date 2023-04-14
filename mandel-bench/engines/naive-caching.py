#!/usr/bin/env python3
from functools import cache


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    @cache
    def mandel(x, y, max_iterations):
        c = complex(x, y)
        c0 = c
        for iteration in range(max_iterations):
            c = c ** 2 + c0
            if abs(c) > 2.0:
                break
        return iteration

    x = [
        x_min + (x_max - x_min) / (resolution - 1) * index
        for index in range(resolution)
    ]
    y = [
        y_min + (y_max - y_min) / (resolution - 1) * index
        for index in range(resolution)
    ]

    iterations = []
    for _y in y:
        row = []
        for _x in x:
            iteration = mandel(_x, abs(_y), max_iterations)
            row.append(iteration)
        iterations.append(row)

    ci = mandel.cache_info()
    details = {"hits": ci.hits, "misses": ci.misses, "size": ci.currsize}
    return iterations, details
