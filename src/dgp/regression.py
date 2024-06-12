from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from dgp import _default_jitter
from dgp.kernels import CovMatrix, derivative_cov_func


class GP(NamedTuple):
    kernel: Callable
    X: Float[Array, "N D"]
    y: Float[Array, "N"]
    L: Float[Array, "N N"]
    alpha: Float[Array, "N 1"]
    cov_matrices: CovMatrix


def fit(
    X: Float[Array, "N D"],
    y: Float[Array, "N"],
    kernel: Callable,
    _jitter: float = _default_jitter,
) -> GP:
    "Fit a GP to data X, y."
    cov_matrices = derivative_cov_func(kernel)

    K = cov_matrices.A(X, X)

    L = jax.scipy.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]), lower=True)
    alpha = jax.scipy.linalg.cho_solve((L, True), y.reshape(-1, 1))

    return GP(kernel, X, y, L, alpha, cov_matrices)


def predict(
    X: Float[Array, "N D"], gp: GP
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    "Compute predictive mean and covariance at locations X."
    k = gp.cov_matrices.B(gp.X, X)

    f_pred = k.T @ gp.alpha
    v = jax.scipy.linalg.solve_triangular(gp.L, k, lower=True)
    covar = gp.cov_matrices.D(X, X) - v.T @ v

    return f_pred, covar


def logp(gp: GP) -> float:
    "Compute the log-marginal likelihood."
    return (
        -0.5 * jnp.dot(gp.y.ravel(), gp.alpha.ravel())
        - jnp.sum(jnp.log(jnp.diag(gp.L)))
        - 0.5 * gp.X.shape[0] * jnp.log(2 * jnp.pi)
    )


def sample(
    key: ArrayLike,
    X: Float[Array, "N D"],
    gp: GP,
    num_samples: int = 1,
    _jitter: float = _default_jitter,
) -> Float[Array, "num_samples N"]:
    "Sample from a GP at locations X."

    _, K = predict(X, gp)

    L = jax.scipy.linalg.cholesky(K + _jitter * jnp.eye(*K.shape), lower=True)
    U = random.normal(key, shape=(num_samples, X.shape[0]))

    samples = jax.vmap(lambda u: L @ u)(U)

    return samples
