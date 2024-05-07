from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import tqdm
from jaxtyping import Array, Float, PyTree

from dgp.kernels import CovMatrix, derivative_cov_func

_jitter = 1e-6


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
    jitter: float = _jitter,
) -> GP:
    cov_matrices = derivative_cov_func(kernel)

    K = cov_matrices.A(X, X)

    L = jax.scipy.linalg.cholesky(K + jitter * jnp.eye(K.shape[0]), lower=True)
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
    return float(
        -0.5 * jnp.dot(gp.y.ravel(), gp.alpha.ravel())
        - jnp.sum(jnp.log(jnp.diag(gp.L)))
        - 0.5 * gp.X.shape[0] * jnp.log(2 * jnp.pi)
    )


DEFAULT_PARAMS = {"lengthscale": jnp.array([2.0, 2.0]), "variance": 2.0}


def tune(
    kernel: Callable, X: Array, y: Array, num_epochs: int, params: dict = DEFAULT_PARAMS
) -> dict:
    def objective(
        params: PyTree,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
    ) -> float:
        k = kernel(**params)
        gp = fit(X, y, k)

        return -logp(gp)

    loss_grad_fn = jax.value_and_grad(objective)

    @jax.jit
    def step_fn(
        params: PyTree,
        opt_state: optax.OptState,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
    ) -> tuple[PyTree, optax.OptState, float]:
        neg_logp, grad = loss_grad_fn(params, X, y)
        update, opt_state = optimiser.update(grad, opt_state)
        params = optax.apply_updates(params, update)
        return params, opt_state, neg_logp

    optimiser = optax.adam(1e-1)
    opt_state = optimiser.init(params)
    logger = {"epoch": [], "logp": []}

    pbar = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for epoch in pbar:
        try:
            params, opt_state, neg_logp = step_fn(params, opt_state, X, y)

            logger["epoch"].append(epoch)
            logger["logp"].append(-neg_logp)

            pbar.set_description(f"Epoch: {epoch}, log marginal: {-neg_logp:g}")

        except KeyboardInterrupt:  # noqa: PERF203
            break

    return params
