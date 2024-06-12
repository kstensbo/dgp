from collections.abc import Callable

import jax
import optax
import tqdm
from jaxtyping import Array, Float, PyTree


def tune(
    kernel: Callable,
    X: Float[Array, "N D"],
    y: Float[Array, "N"],
    num_epochs: int,
    params: PyTree,
) -> PyTree:
    "Convenience function for training a GP."

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
