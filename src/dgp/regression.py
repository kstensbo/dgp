from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from dgp.kernels import derivative_cov_func, CovMatrix

_jitter = 1e-6


class GP(NamedTuple):
    kernel: Callable
    X: Float[Array, "N D"]
    y: Float[Array, "N"]
    L: Float[Array, "N N"]
    alpha: Float[Array, "N 1"]
    cov_matrices: CovMatrix


def fit(X: Float[Array, "N D"], y: Float[Array, "N"], kernel: Callable) -> GP:
    cov_matrices = derivative_cov_func(kernel)

    K = cov_matrices.A(X, X)

    L = jax.scipy.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]), lower=True)
    alpha = jax.scipy.linalg.cho_solve((L, True), y.reshape(-1, 1))

    return GP(kernel, X, y, L, alpha, cov_matrices)


def predict(
    X: Float[Array, "N D"], gp: GP
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    k = gp.cov_matrices.B(gp.X, X)

    f_pred = k.T @ gp.alpha
    v = jax.scipy.linalg.solve_triangular(gp.L, k, lower=True)
    covar = gp.cov_matrices.D(X, X) - v.T @ v

    return f_pred, covar


def logp(gp: GP) -> float:
    return (
        -0.5 * jnp.inner(gp.y, gp.alpha)
        - jnp.sum(jnp.log(jnp.diag(gp.L)))
        - 0.5 * gp.X.shape[0] * jnp.log(2 * jnp.pi)
    )
