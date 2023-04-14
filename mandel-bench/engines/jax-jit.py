#!/usr/bin/env python3
import functools
import jax.numpy as jnp
from jax import jit


def calculate(x_min, x_max, y_min, y_max, max_iterations, resolution):
    @functools.partial(jit, static_argnums=1)
    def mandel(c, max_iterations):
        c0 = c.copy()
        iterations = jnp.zeros_like(c, dtype=jnp.uint32)

        for iteration in range(max_iterations):
            mask = jnp.abs(c) < 2
            c = jnp.where(mask, c ** 2 + c0, c)
            iterations = jnp.where(mask, iteration, iterations)
        return iterations

    x = jnp.linspace(x_min, x_max, resolution)
    y = jnp.linspace(y_min, y_max, resolution)

    c = x + y[:, None] * 1j
    iterations = mandel(c, max_iterations)

    return iterations, {}
