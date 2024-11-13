import argparse
import json
import pathlib
from collections.abc import Callable
from typing import NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import prettytable
import tqdm
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, PyTree
from matplotlib.colors import CenteredNorm

from dgp import kernels
from dgp.settings import _default_jitter

jax.config.update("jax_enable_x64", True)


class Dataset(NamedTuple):
    X: Array
    Y: Array


class GP(NamedTuple):
    k: Callable
    data: Dataset
    K: Array
    L: Array
    alpha: Array


class GPPredictions(NamedTuple):
    X: Array
    mean: Array
    variance: Array


class Logger(TypedDict):
    epochs: list[int]
    values: list[float]


def sample_dataset(
    key: ArrayLike,
    params: PyTree,
    num_samples: int,
) -> Dataset:
    key, subkey = random.split(key)

    N = num_samples
    D = params["log_lengthscale"].shape[0]

    # Sample the input coordinates:
    X = random.uniform(subkey, (N // 2, D), minval=-1, maxval=0)
    X = jnp.concatenate(
        [X, random.uniform(subkey, (N // 2, D), minval=0, maxval=1)], axis=0
    )

    # Define a divergence-free GP and sample outputs:
    base_kernel = kernels.eq(
        lengthscale=jnp.exp(params["log_lengthscale"]),
        variance=jnp.exp(params["log_variance"]),
    )
    k_df = kernels.div_free(base_kernel)

    # Covariance tensor, shape [N, N, D, D]:
    C = kernels.cov_matrix(k_df, X, X)

    # Covariance matrix, shape [D*N, D*N]:
    K = kernels.tensor_to_matrix(C)

    # Sample from the GP:
    key, subkey = random.split(key)
    L = jnp.linalg.cholesky(
        K
        + (jnp.exp(params["log_likelihood_variance"]) + _default_jitter)
        * jnp.eye(K.shape[0])
    )
    u = random.normal(subkey, shape=[K.shape[0], 1])
    F = L @ u

    # Reshape to recover vectors:
    F = F.squeeze().reshape(N, D)

    return Dataset(X=X, Y=F)


def construct_div_free_gp(params: PyTree, training_set: Dataset) -> GP:
    # Define a divergence-free GP:
    base_kernel = kernels.eq(
        lengthscale=jnp.exp(params["log_lengthscale"]),
        variance=jnp.exp(params["log_variance"]),
    )
    k_df = kernels.div_free(base_kernel)

    # Covariance tensor, shape [N, N, D, D]:
    C = kernels.cov_matrix(k_df, training_set.X, training_set.X)

    # Covariance matrix, shape [D*N, D*N]:
    K = kernels.tensor_to_matrix(C)

    # Compute and return the marginal log-likelihood:
    L = jnp.linalg.cholesky(
        K
        + (jnp.exp(params["log_likelihood_variance"]) + _default_jitter)
        * jnp.eye(K.shape[0])
    )
    alpha = jax.scipy.linalg.cho_solve((L, True), training_set.Y.reshape(-1, 1))

    return GP(k=k_df, data=training_set, K=K, L=L, alpha=alpha)


def fit_gp(
    params: PyTree,
    training_set: Dataset,
    optimiser: optax.GradientTransformation,
    num_epochs: int,
) -> tuple[PyTree, Logger]:
    "Fit a GP to data."

    def objective(params: PyTree, data: Dataset) -> float:
        "Compute the log marginal likelihood."
        gp = construct_div_free_gp(params, data)

        logp = (
            -0.5 * jnp.dot(gp.data.Y.reshape(-1, 1).ravel(), gp.alpha.ravel())
            - jnp.sum(jnp.log(jnp.diag(gp.L)))
            - 0.5 * gp.K.shape[0] * jnp.log(2 * jnp.pi)
        )

        return -logp

    loss_grad_fn = jax.value_and_grad(objective)

    @jax.jit
    def step_fn(
        params: PyTree, opt_state: optax.OptState, data: Dataset
    ) -> tuple[PyTree, optax.OptState, float]:
        "Step function for the optimiser."
        neg_logp, grad = loss_grad_fn(params, data)
        update, opt_state = optimiser.update(grad, opt_state)
        params = optax.apply_updates(params, update)
        return params, opt_state, -neg_logp

    logger = Logger(epochs=[], values=[])
    pbar = tqdm.tqdm(range(num_epochs), desc="Epoch")

    opt_state = optimiser.init(params)
    try:
        for epoch in pbar:
            params, opt_state, logp = step_fn(params, opt_state, training_set)

            logger["epochs"].append(epoch)
            logger["values"].append(logp)

            pbar.set_description(f"Epoch {epoch:d}: log(p) = {logp:.4g}")

    except KeyboardInterrupt:
        print("Training aborted.")

    return params, logger


def predict(gp: GP, X: Array) -> GPPredictions:
    "Compute GP posterior predictions at locations X."

    # Covariance tensors:
    Cxs = kernels.cov_matrix(gp.k, gp.data.X, X)  # [N, M, D, D]
    Cxx = kernels.cov_matrix(gp.k, X, X)  # [M, M, D, D]

    # Covariance matrices:
    Kxs = kernels.tensor_to_matrix(Cxs)  # [D*N, D*M]
    Kxx = kernels.tensor_to_matrix(Cxx)  # [D*M, D*M]

    f_pred = Kxs.T @ gp.alpha
    v = jax.scipy.linalg.solve_triangular(gp.L, Kxs, lower=True)
    covar = Kxx - v.T @ v
    var = jnp.diag(covar)

    M = X.shape[0]
    D = gp.data.Y.shape[1]

    return GPPredictions(X=X, mean=f_pred.reshape(M, D), variance=var.reshape(M, D))


def fit_and_predict(
    params: PyTree,
    training_set: Dataset,
    X_test: Array,
    optimiser: optax.GradientTransformation,
    num_epochs: int,
) -> tuple[PyTree, GPPredictions, Logger]:
    "Fit a GP to training_set and compute predictions at X_test."

    params, logger = fit_gp(
        params=params,
        training_set=training_set,
        optimiser=optimiser,
        num_epochs=num_epochs,
    )

    # Construct a GP and precompute matrices for faster prediction:
    gp = construct_div_free_gp(params, training_set)

    predictions = predict(gp, X_test)

    return params, predictions, logger


def get_optimiser(name: str, lr: float) -> optax.GradientTransformation:
    try:
        optimiser = getattr(optax, name)

    except AttributeError as atr:
        msg = f"Optimiser {name} unknown."
        raise ValueError(msg) from atr

    else:
        return optimiser(learning_rate=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=32, help="Number of training samples."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="Number of training epochs."
    )
    parser.add_argument(
        "--optimiser", type=str, default="adam", help="Name of Optax optimiser."
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument("--data", choices=["random", "synthetic"], default="synthetic")

    args = parser.parse_args()

    key = random.PRNGKey(args.seed)

    if args.data == "random":
        true_params = {
            "log_lengthscale": jnp.log(jnp.ones(2) * 0.5),
            "log_variance": jnp.log(2.0),
            "log_likelihood_variance": jnp.log(0.1),
        }
        training_set = sample_dataset(
            key=key,
            params=true_params,
            num_samples=args.num_samples,
        )

    elif args.data == "synthetic":
        with pathlib.Path("data/training_set.json").open("r") as f:
            training_data = json.load(f)

        training_set = Dataset(
            X=jnp.array(training_data["X"]), Y=jnp.array(training_data["Y"])
        )

    else:
        msg = f"Value of data parameter unknown: '{args.data}'."
        raise ValueError(msg)

    optimiser = get_optimiser(args.optimiser, lr=args.lr)

    # Set initial parameters:
    initial_params = {
        "log_lengthscale": jnp.zeros(2),
        "log_variance": jnp.log(1.0),
        "log_likelihood_variance": jnp.log(1.0),
    }

    # Define prediction locations:
    Nx = 64
    Ny = 64
    N = Nx * Ny

    x_array = jnp.linspace(-1.5, 1.5, Nx)
    y_array = jnp.linspace(-1.5, 1.5, Ny)

    xx, yy = jnp.meshgrid(x_array, y_array, indexing="ij")
    X_test = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

    params, predictions, logger = fit_and_predict(
        params=initial_params,
        training_set=training_set,
        X_test=X_test,
        optimiser=optimiser,
        num_epochs=args.num_epochs,
    )

    # Pretty print results of optimisation:
    results_table = prettytable.PrettyTable()
    results_table.field_names = [
        "Parameter",
        "Length scales",
        "Variance",
        "Likelihood variance",
    ]
    if args.data == "random":
        results_table.add_row(
            [
                "True",
                jnp.exp(true_params["log_lengthscale"]),
                float(jnp.exp(true_params["log_variance"])),
                float(jnp.exp(true_params["log_likelihood_variance"])),
            ]
        )

    results_table.add_row(
        [
            "Found",
            jnp.exp(params["log_lengthscale"]),
            float(jnp.exp(params["log_variance"])),
            float(jnp.exp(params["log_likelihood_variance"])),
        ]
    )

    results_table.float_format = "5.3"

    print(results_table)

    # Compute divergence:
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]
    f_grid = predictions.mean.reshape(Nx, Ny, 2)

    dfdx = jnp.gradient(f_grid[..., 0], dx, axis=0)
    dfdy = jnp.gradient(f_grid[..., 1], dy, axis=1)

    divergence = dfdx + dfdy

    _, ax = plt.subplot_mosaic(
        """
        AAABB
        AAABB
        AAACC
        """,
        figsize=(10, 5),
        layout="constrained",
    )

    ax["A"].quiver(
        *training_set.X.T,
        *training_set.Y.T,
        angles="xy",
        color="C0",
        label="Training set",
        alpha=0.7,
        zorder=10,
    )

    ax["A"].quiver(
        *predictions.X.T,
        *predictions.mean.T,
        angles="xy",
        color="C1",
        label="Predictions",
        zorder=5,
    )

    c_var = ax["A"].imshow(
        predictions.variance.reshape(Nx, Ny, 2).sum(-1),
        extent=[-1.5, 1.5, -1.5, 1.5],
        origin="lower",
        label="Summed variance",
        cmap="Oranges",
        zorder=0,
    )
    plt.colorbar(c_var, label="Summed variance")

    c_div = ax["B"].imshow(
        divergence,
        extent=[-1.5, 1.5, -1.5, 1.5],
        origin="lower",
        cmap="coolwarm",
        norm=CenteredNorm(),
    )
    plt.colorbar(c_div, label="Divergence")

    ax["C"].plot(
        logger["epochs"],
        logger["values"],
        ls="-",
        label="Log marginal likelihood",
    )

    ax["A"].set_title("Field")
    ax["A"].legend()

    ax["B"].set_title("Divergence")

    ax["C"].set_title("Log marginal likelihood")
    ax["C"].set_xlabel("Epoch")

    plt.show()
    # plt.savefig("divergence_free_field.png", dpi=300)
