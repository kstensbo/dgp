from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def cov_matrix(
    kernel: Callable, x: Float[Array, "N D"], y: Float[Array, "M D"]
) -> Float[Array, "N M"]:
    "Compute a dense covariance matrix from a kernel function and arrays of inputs."

    K = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(y))(x)

    return K


def eq(lengthscale: Float[Array, "D"], variance: float) -> Callable:
    "The exponentiated quadratic covariance function."

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
        return variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))

    return k


def eq_kernel(params: PyTree, x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
    variance = params["variance"]
    lengthscale = params["lengthscale"]
    return variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))


class CovMatrix(NamedTuple):
    A: Callable
    B: Callable
    C: Callable
    D: Callable


def derivative_cov_func(kernel: Callable) -> CovMatrix:
    # Assume the following covariance matrix block structure,
    #
    #   | A   B |
    #   | C   D |
    #
    # where A is the Gram matrix of the derivative observations, D is the Gram matrix of
    # the actual observations, and C.T == B.
    #

    # ===== Constructing the covariance function for A =====
    # This derivative function outputs a [D, D] matrix for a pair of inputs:
    d2kdxdy = jax.jacfwd(jax.jacrev(kernel, argnums=0), argnums=1)

    # vmap over each input to the covariance function in turn. We want each argument to
    # keep their mutual order in the resulting matrix, hence the out_axes
    # specifications:
    d2kdxdy_cov = jax.vmap(
        jax.vmap(d2kdxdy, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0),
        out_axes=1,
    )

    def ensure_2d_function_values(
        f: Float[Array, "M"] | Float[Array, "M 1"],
    ) -> Float[Array, "M 1"]:
        "Make sure function values have shape [M 1]."
        return jax.lax.cond(
            len(f.shape) == 1, lambda x: jnp.atleast_2d(x).T, lambda x: x, f
        )

    def A(dx: Float[Array, "N D"], dy: Float[Array, "N D"]) -> Float[Array, "N N"]:
        batched_matrix = d2kdxdy_cov(dx, dy)
        matrix = jnp.concatenate(jnp.concatenate(batched_matrix, axis=1), axis=1)
        return matrix

    # ===== Constructing the covariance function for B =====

    ddx = jax.jacfwd(kernel, argnums=0)
    ddx_cov = jax.vmap(
        jax.vmap(ddx, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0),
        out_axes=2,
    )

    def B(dx: Float[Array, "N D"], y: Float[Array, "M 1"]) -> Float[Array, "N M"]:
        batched_matrix = ddx_cov(dx, y)
        matrix = jnp.concatenate(batched_matrix, axis=0)
        return matrix

    # ===== Constructing the covariance function for C =====

    ddy = jax.jacfwd(kernel, argnums=1)
    ddy_cov = jax.vmap(
        jax.vmap(ddy, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0),
        out_axes=0,
    )

    def C(x: Float[Array, "M 1"], dy: Float[Array, "N D"]) -> Float[Array, "M N"]:
        batched_matrix = ddy_cov(x, dy)
        matrix = jnp.concatenate(batched_matrix, axis=1)
        return matrix

    # ===== Constructing the covariance function for D =====
    # This is just the covariance function itself.

    def D(x: Float[Array, "M 1"], y: Float[Array, "M 1"]) -> Float[Array, "M M"]:
        return cov_matrix(kernel, x, y)

    return CovMatrix(A, B, C, D)
