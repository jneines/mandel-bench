#!/usr/bin/env python3


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    def mandelbrot(x_min, x_max, y_min, y_max, max_iterations, resolution):
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
                c = complex(_x, _y)
                c0 = c
                for iteration in range(max_iterations):
                    c = c ** 2 + c0
                    if abs(c) > 2.0:
                        break
                row.append(iteration)
            iterations.append(row)
        return iterations

    iterations = mandelbrot(x_min, x_max, y_min, y_max, max_iterations, resolution)
    return iterations, {}
