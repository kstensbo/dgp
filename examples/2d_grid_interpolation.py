import argparse
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PyTree

import dgp.regression as gpr
from dgp import _default_jitter
from dgp.kernels import cov_matrix, eq

from IPython import embed  # noqa

# Set JAX to use 64bit:
jax.config.update("jax_enable_x64", True)


class Dataset(NamedTuple):
    X: Array
    y: Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--resx", type=int, default=64, help="Surface resolution in x direction."
    )
    parser.add_argument(
        "--resy", type=int, default=32, help="Surface resolution in y direction."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Number of training epochs."
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        default="adam",
        help="Optimiser (choose any defined in optax).",
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")

    args = parser.parse_args()
    return args


def generate_toydata(
    key: ArrayLike,
    x_array: ArrayLike,
    y_array: ArrayLike,
    _jitter: float = _default_jitter,
) -> tuple[
    Dataset,
    Dataset,
]:
    xx, yy = jnp.meshgrid(x_array, y_array)

    # Construct input coordinates for grid, shape [N, 2]:
    X = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
    K = cov_matrix(k, X, X)
    L = jnp.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]))

    # Sample noise:
    u = random.normal(key, shape=[X.shape[0], 1])

    # Transform noise to GP prior sample:
    f = L @ u

    f_grid = f.reshape(y_array.shape[0], x_array.shape[0])
    f_grid = jnp.flipud(f_grid)

    # Function derivative:
    dfdy, dfdx = jnp.gradient(f_grid)
    df = jnp.dstack((dfdx, dfdy)).reshape(-1, 2)

    function_data = Dataset(X=X, y=f_grid)
    function_derivative = Dataset(X=X, y=df)

    return function_data, function_derivative


def construct_training_set(
    key: ArrayLike,
    data: Dataset,
    # X: Float[Array, "N D"],
    # df: Float[Array, "N D"],
    train_frac: float,
    layout: str = "random",
) -> tuple[Float[Array, "Ntrain D"], Float[Array, "Ntrain D"]]:
    "Construct training set as a subset of all data."

    num_data = data.X.shape[0]

    # Define indices of full dataset to use for training:

    if layout == "random":
        train_idx = random.choice(
            key,
            num_data,
            shape=(int(num_data * train_frac),),
            replace=False,
        )

    elif layout == "circle":
        # Attempt at using training observations from a circle:
        c = data.X.mean(axis=0)
        r = 1.5
        theta = jnp.linspace(0, 2 * jnp.pi, 20)
        xc = r * jnp.cos(theta) + c[0]
        yc = r * jnp.sin(theta) + c[1]

        Xb = jnp.dstack((xc, yc)).squeeze()

        idx = []
        for i in range(Xb.shape[0]):
            min_idx = jnp.argmin(jnp.sum((data.X - Xb[i]) ** 2, axis=1))
            idx.append(min_idx)

        train_idx = jnp.array(idx)

    else:
        msg = f"Layout {layout} must be 'random' or 'circle'."
        raise ValueError(msg)

    # Create training set as subset of data:
    Xtrain = data.X[train_idx]
    dftrain = data.y[train_idx]

    return Dataset(X=Xtrain, y=dftrain)


def get_optimiser(name: str, lr: float) -> optax.GradientTransformation:
    try:
        optimiser = getattr(optax, name)

    except AttributeError as atr:
        msg = f"Optimiser {name} unknown."
        raise ValueError(msg) from atr

    else:
        return optimiser(learning_rate=lr)


def optimise_hyperparameters(
    params: PyTree,
    kernel: Callable,
    optimiser: optax.GradientTransformation,
    data: Dataset,
    num_epochs: int,
) -> tuple[PyTree, dict[str, list[float]]]:
    "Optimise hyperparameters of the covariance function."

    opt_state = optimiser.init(params)

    def objective(
        params: PyTree,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
    ) -> float:
        k = kernel(
            lengthscale=jnp.exp(params["log_lengthscale"]), variance=params["variance"]
        )
        gp = gpr.fit(X, y, k)

        return -gpr.logp(gp)

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

    logger = {
        "epoch": [],
        "logp": [],
    }
    pbar = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for epoch in pbar:
        try:
            params, opt_state, neg_logp = step_fn(params, opt_state, data.X, data.y)

            logger["epoch"].append(epoch)
            logger["logp"].append(-neg_logp)

            pbar.set_description(f"Epoch: {epoch}, log marginal: {-neg_logp:g}")

        except KeyboardInterrupt:
            break

    return params, logger


def compute_cosine_similarity(
    f_grad: Float[Array, "N M"], f_pred: Float[Array, "N M"]
) -> Float[Array, "N M"]:
    "Compute cosine similarity between predicted and true gradients"

    dfdy_pred, dfdx_pred = jnp.gradient(f_pred)
    df_pred = jnp.dstack((dfdx_pred, dfdy_pred)).reshape(-1, 2)

    # Cosine similarity between predicted and true gradients:
    cosine_similarity = jnp.sum(f_grad * df_pred, axis=1) / jnp.sqrt(
        jnp.sum(f_grad**2, axis=1) * jnp.sum(df_pred**2, axis=1)
    )
    cosine_similarity = cosine_similarity.reshape(*f_pred.shape)

    return cosine_similarity


def main() -> None:
    args = parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # Set up toy dataset:
    x_array = jnp.linspace(0, 10, args.resx)
    y_array = jnp.linspace(0, 5, args.resy)

    function_data, function_derivative = generate_toydata(subkey, x_array, y_array)

    key, subkey = random.split(key)
    training_set = construct_training_set(subkey, function_derivative, train_frac=0.03)

    # ==================== Optimise the GP ====================

    # Define the covariance function to use and the initial hyperparameters:
    kernel = eq
    params = {
        "log_lengthscale": jnp.array([1.0, 1.0]),
        "variance": 2.0,
    }

    optimiser = get_optimiser(args.optimiser, args.lr)

    # Optimise!
    params, logger = optimise_hyperparameters(
        params=params,
        kernel=kernel,
        optimiser=optimiser,
        data=training_set,
        num_epochs=args.num_epochs,
    )

    print("Optimised parameters:")
    for key, val in params.items():
        print(f"  {key} = {val}")

    # Use the optimised parameters:
    k = eq(lengthscale=jnp.exp(params["log_lengthscale"]), variance=params["variance"])
    gp = gpr.fit(training_set.X, training_set.y, k)

    f_pred, covar = gpr.predict(function_data.X, gp)
    std = jnp.sqrt(jnp.diag(covar))

    # Reshape to grid and compute gradients:
    f_pred = f_pred.reshape(args.resy, args.resx)
    std = std.reshape(args.resy, args.resx)

    cosine_similarity = compute_cosine_similarity(function_derivative.y, f_pred)

    # ==================== Plotting ====================
    _, ax = plt.subplots(
        3,
        2,
        figsize=(12, 10),
        layout="constrained",
    )

    grid_extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]

    true_c = ax[0, 0].imshow(
        function_data.y,
        extent=grid_extent,
        origin="upper",
    )

    ax[0, 0].scatter(*training_set.X.T, s=12, facecolor=plt.cm.Oranges(0.5), alpha=1)

    ax[0, 1].quiver(
        *function_derivative.X.T,
        *function_derivative.y.T,
        function_data.y.ravel(),
        angles="xy",
    )

    pred_c = ax[1, 0].imshow(
        f_pred,
        extent=grid_extent,
        origin="upper",
    )
    ax[1, 0].scatter(*training_set.X.T, s=12, facecolor=plt.cm.Oranges(0.5), alpha=1)

    diff_c = ax[1, 1].imshow(
        cosine_similarity,
        extent=grid_extent,
        origin="upper",
        cmap="Spectral",
        vmin=-1,
        vmax=1,
    )

    ax[2, 0].plot(
        logger["epoch"], logger["logp"], c="C0", label="Marginal log-likelihood"
    )

    var_c = ax[2, 1].imshow(
        std,
        extent=grid_extent,
        origin="upper",
        cmap="Purples",
        vmin=0,
    )

    ax[0, 1].set_xlim(x_array[0], x_array[-1])
    ax[0, 1].set_ylim(y_array[0], y_array[-1])
    ax[0, 1].set_ylim(y_array[-1], y_array[0])
    ax[0, 1].set_aspect("equal")

    ax[2, 0].set_ylim(bottom=-100, top=300)

    plt.colorbar(true_c, ax=ax[0, 0])
    plt.colorbar(pred_c, ax=ax[1, 0])
    plt.colorbar(diff_c, ax=ax[1, 1])
    plt.colorbar(var_c, ax=ax[2, 1])

    ax[0, 0].set_title("True function")
    ax[0, 1].set_title("Gradient field")
    ax[1, 0].set_title("Predicted function")
    ax[1, 1].set_title("Cosine similarity of gradients")
    ax[2, 0].set_title("Marginal log-likelihood")
    ax[2, 1].set_title("Predictive uncertainty on function values (1 std)")

    ax[2, 0].set_xlabel("Epoch")
    ax[2, 0].set_ylabel("Marginal log-likelihood")

    plt.show()
    # plt.savefig("gp_on_derivative_obs_toy_data.pdf")


if __name__ == "__main__":
    main()
