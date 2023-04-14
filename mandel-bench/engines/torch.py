#!/usr/bin/env python3
import torch

print(torch.cuda.is_available())
torch.device("cuda")


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    x = torch.linspace(x_min, x_max, resolution)
    y = torch.linspace(y_min, y_max, resolution)

    c = x + y[:, None] * 1j
    c0 = c.clone().detach()
    iterations = torch.zeros_like(c, dtype=torch.int32)

    for iteration in range(max_iterations):
        mask = torch.abs(c) < 2
        c[mask] = c[mask] ** 2 + c0[mask]

        iterations[mask] += 1

    return iterations.detach().numpy(), {}
