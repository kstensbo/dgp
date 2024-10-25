from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar


def cov_matrix(
    kernel: Callable, x: Float[Array, "N D"], y: Float[Array, "M D"]
) -> Float[Array, "N M"] | Float[Array, "N M P P"]:
    "Compute a dense covariance matrix from a kernel function and arrays of inputs."

    # K = jax.vmap(lambda xi: jax.vmap(lambda yi: kernel(xi, yi))(y))(x)
    # FIXME: Should the input be (y, x) instead of (x, y)?
    K = jax.vmap(
        jax.vmap(kernel, in_axes=(0, None), out_axes=0),
        in_axes=(None, 0),
        out_axes=1,
    )(x, y)

    return K


def tensor_to_matrix(tensor: Float[Array, "N M P Q"]) -> Float[Array, "N*P M*Q"]:
    "Take an [N, M, P, Q] tensor and transform it into an [N*P, M*Q] tiled matrix."
    N, M, P, Q = tensor.shape
    return jnp.transpose(tensor, axes=(0, 2, 1, 3)).reshape((N * P, M * Q))


def eq(lengthscale: Float[Array, "D"], variance: Scalar | float) -> Callable:
    "The exponentiated quadratic covariance function."

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Scalar:
        return variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))

    return k


def ess(lengthscale: Float[Array, "D"], variance: float, period: float) -> Callable:
    "The exponential sine square (periodic) covariance function."

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
        X = jnp.sin(jnp.pi * jnp.abs(x - y) / period) / lengthscale
        return variance * jnp.exp(-2.0 * X.dot(X))

    return k


def levi_civita(i: int, j: int, k: int) -> int:
    "Computes the Levi-Civita symbol for a 3D tensor."
    if (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
        symbol = 1
    elif (i, j, k) in [(3, 2, 1), (1, 3, 2), (2, 1, 3)]:
        symbol = -1
    else:
        symbol = 0

    return symbol


def div_free(base_kernel: Callable | Sequence[Sequence[Callable]]) -> Callable:
    """
    Defines a divergence-free kernel. Assumes the same base kernel for all elements in
    the matrix-valued kernel.
    """

    # List of non-zero Levi-Civita symbols:
    nonzero_levi_civita_symbols = [
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, 2),
        (3, 2, 1),
        (1, 3, 2),
        (2, 1, 3),
    ]

    if callable(base_kernel):
        # This derivative function outputs a [D, D] matrix for a pair of inputs:
        # for i in range(3):
        #     row = []
        #     for j in range(3):
        #         row.append(jax.jacfwd(jax.jacrev(base_kernel, argnums=0), argnums=1))
        #
        #     d2kdxdy.append(row)

        d2kdxdy = [
            [
                jax.jacfwd(jax.jacrev(base_kernel, argnums=0), argnums=1)
                for _ in range(3)
            ]
            for _ in range(3)
        ]

    else:
        d2kdxdy = [
            [
                jax.jacfwd(jax.jacrev(base_kernel[i][j], argnums=0), argnums=1)
                for j in range(3)
            ]
            for i in range(3)
        ]

    # vmap over each input to the covariance function in turn. We want each argument to
    # keep their mutual order in the resulting matrix, hence the out_axes
    # specifications:
    # d2kdxdy_cov = jax.vmap(
    #     jax.vmap(d2kdxdy, in_axes=(0, None), out_axes=0),
    #     in_axes=(None, 0),
    #     out_axes=1,
    # )

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "D D"]:
        K = jnp.zeros([3, 3], dtype=float)
        for i, k, l in nonzero_levi_civita_symbols:  # noqa: E741
            for j, m, n in nonzero_levi_civita_symbols:
                K = K.at[i - 1, j - 1].add(
                    levi_civita(i, k, l)
                    * levi_civita(j, m, n)
                    * d2kdxdy[l - 1][n - 1](x, y)[k - 1, m - 1]
                )  # [l, n]

        return K

    return k


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

    # def ensure_2d_function_values(
    #     f: Float[Array, "M"] | Float[Array, "M 1"],
    # ) -> Float[Array, "M 1"]:
    #     "Make sure function values have shape [M 1]."
    #     return jax.lax.cond(
    #         len(f.shape) == 1, lambda x: jnp.atleast_2d(x).T, lambda x: x, f
    #     )

    def A(dx: Float[Array, "N D"], dy: Float[Array, "N D"]) -> Float[Array, "D*N D*N"]:
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

    def B(dx: Float[Array, "N D"], y: Float[Array, "M D"]) -> Float[Array, "D*N M"]:
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

    def C(x: Float[Array, "M D"], dy: Float[Array, "N D"]) -> Float[Array, "M D*N"]:
        batched_matrix = ddy_cov(x, dy)
        matrix = jnp.concatenate(batched_matrix, axis=1)
        return matrix

    # ===== Constructing the covariance function for D =====
    # This is just the covariance function itself.

    def D(x: Float[Array, "M D"], y: Float[Array, "M D"]) -> Float[Array, "M M"]:
        return cov_matrix(kernel, x, y)

    return CovMatrix(A, B, C, D)
