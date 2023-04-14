#!/usr/bin/env python3
from functools import cache


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    def force_symmetry(y):
        yp = [_y for _y in y if _y >= 0]
        yn = [_y for _y in y if _y < 0]

        if len(yp) >= len(yn):
            # there are more positive than negative numbers
            # invert positives to create new negatives
            # but start with the right index
            new_y = [-v for v in yp[::-1]]
            new_y.extend(yp)
            return new_y[len(yp) - len(yn) :]
        else:
            # there are more negative than positive numbers
            # invert negatives to create new positives
            # but only go to the

            new_y = yn
            new_y.extend([-_y for _y in yn[::-1]])
            return new_y[: len(y)]

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
    y = force_symmetry(y)

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
