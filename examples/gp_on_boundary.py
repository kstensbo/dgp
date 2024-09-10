import argparse
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PyTree
from matplotlib.colors import Normalize

import dgp.regression as gpr
from dgp.kernels import cov_matrix, eq
from dgp.settings import _default_jitter

from IPython import embed  # noqa

# Set JAX to use 64bit:
jax.config.update("jax_enable_x64", True)


class Dataset(NamedTuple):
    X: Array
    y: Array
    dy: Array
    resx: float
    resy: float


class CircleDataset(NamedTuple):
    X: Array
    f: Array  # Latent function
    y: Array  # Observed data
    theta: Array


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
        "--num_training_data", type=int, default=32, help="Number of training data."
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
    parser.add_argument(
        "--scatter",
        type=str,
        choices=["random", "even"],
        default="even",
        help="Distribution of data along the circle.",
    )

    args = parser.parse_args()
    return args


def generate_toydata(
    key: ArrayLike,
    mesh: ArrayLike,
    num_training_data: int,
    num_test_data: int = 256,
    data_scatter: str = "even",
    _jitter: float = _default_jitter,
) -> tuple[
    Dataset,
    CircleDataset,
    CircleDataset,
]:
    # Construct input coordinates for grid, shape [N, 2]:
    X_grid = jnp.stack(mesh, axis=-1).reshape(-1, 2)

    # For circle data:
    centre = jnp.array([mesh[0].mean(), mesh[1].mean()])
    radius = 1.5

    if data_scatter == "random":
        # Use randomly selected points on the circle:
        key, subkey = random.split(key)
        theta = random.uniform(
            subkey, shape=(num_training_data, 1), minval=0, maxval=2 * jnp.pi
        ).sort()
    elif data_scatter == "even":
        # Use evenly spaced points on the circle:
        theta = jnp.linspace(0, 2 * jnp.pi, num_training_data)[:, None]

    else:
        err_msg = f"Unknown data_scatter value: {data_scatter}"
        raise ValueError(err_msg)

    points_x = radius * jnp.cos(theta) + centre[0]
    points_y = radius * jnp.sin(theta) + centre[1]

    X_train = jnp.concatenate([points_x, points_y], axis=1)

    # Test points:
    theta_test = jnp.linspace(0, 2 * jnp.pi, num_test_data)
    points_x_test = radius * jnp.cos(theta_test) + centre[0]
    points_y_test = radius * jnp.sin(theta_test) + centre[1]
    # X_test = jnp.concatenate([points_x_test, points_y_test], axis=1)
    X_test = jnp.stack([points_x_test, points_y_test], axis=-1)

    # # For a uniform distribution:
    # X_train = random.uniform(subkey[1], shape=(num_training_data, 2))
    # X_train = X_train.at[:, 0].set(
    #     X_train[:, 0] * (x_array[-1] - x_array[0]) + x_array[0]
    # )
    # X_train = X_train.at[:, 1].set(
    #     X_train[:, 1] * (y_array[-1] - y_array[0]) + y_array[0]
    # )

    # Additional locations for computing gradients at training locations:
    gradient_epsilon = 5e-1  # Resolution for finite differences

    # Compute grid around each training point and flatten:
    grad_x = jnp.linspace(-gradient_epsilon, gradient_epsilon, 3)
    grad_y = jnp.linspace(-gradient_epsilon, gradient_epsilon, 3)
    grad_mesh = jnp.array(jnp.meshgrid(grad_x, grad_y))
    X_train_mesh = grad_mesh + X_train[..., None, None]
    X_grad = jnp.rollaxis(X_train_mesh, 1, 4).reshape(-1, 2)

    X = jnp.concatenate((X_grid, X_train, X_test, X_grad), axis=0)

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
    K = cov_matrix(k, X, X)
    L = jnp.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]))

    # Sample noise:
    u = random.normal(key, shape=[X.shape[0], 1])

    # Transform noise to GP prior sample:
    f = L @ u

    f_grid, f_train, f_test, f_grad = jnp.split(
        f, jnp.array([X_grid.shape[0], X_train.shape[0], X_test.shape[0]]).cumsum()
    )

    f_grid = f_grid.reshape(*mesh[0].shape)

    # Function derivative:
    dfdy, dfdx = jnp.gradient(f_grid)
    df_grid = jnp.dstack((dfdx, dfdy)).reshape(-1, 2)

    # Function derivative at training locations:
    f_grad = f_grad.reshape(-1, 3, 3)
    df_train_dy, df_train_dx = jnp.gradient(
        f_grad, gradient_epsilon, gradient_epsilon, axis=(1, 2)
    )
    df_train = jnp.dstack((df_train_dx[:, 1, 1], df_train_dy[:, 1, 1])).reshape(-1, 2)
    assert df_train.shape == (num_training_data, 2)

    function_data = Dataset(
        X=X_grid,
        y=f_grid,
        dy=df_grid,
        resx=mesh[0][0, 1] - mesh[0][0, 0],
        resy=mesh[1][1, 0] - mesh[1][0, 0],
    )

    training_set = CircleDataset(
        X=X_train,
        f=f_train,
        y=df_train,
        theta=theta,
    )

    test_set = CircleDataset(X=X_test, f=f_test, y=jnp.array([]), theta=theta_test)

    return function_data, training_set, test_set


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
    data: Dataset | CircleDataset,
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
            lengthscale=jnp.exp(params["log_lengthscale"]),
            variance=jnp.exp(params["log_variance"]),
        )
        gp = gpr.fit(X, y, k)

        return -gpr.logp(gp)

    loss_grad_fn = jax.value_and_grad(objective)

    @jax.jit
    def step_fn(
        params: PyTree,
        opt_state: optax.OptState,
        X: Float[Array, "N Dx"],
        y: Float[Array, "N Dy"],
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
    try:
        for epoch in pbar:
            params, opt_state, neg_logp = step_fn(params, opt_state, data.X, data.y)

            logger["epoch"].append(epoch)
            logger["logp"].append(-neg_logp)

            pbar.set_description(f"Epoch: {epoch}, log marginal: {-neg_logp:g}")

    except KeyboardInterrupt:
        print("Training interrupted.")

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


def main() -> None:  # noqa: PLR0915
    args = parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # Set up toy dataset:
    x_array = jnp.linspace(0, 10, args.resx)
    y_array = jnp.linspace(0, 5, args.resy)
    mesh = jnp.meshgrid(x_array, y_array)

    function_data, training_set, test_set = generate_toydata(
        subkey,
        mesh,
        num_training_data=args.num_training_data,
        data_scatter=args.scatter,
    )

    # ==================== Optimise the GP ====================

    # Define the covariance function to use and the initial hyperparameters:
    kernel = eq
    params = {
        "log_lengthscale": jnp.array([0.0, 0.0]),
        "log_variance": 0.0,
    }

    optimiser = get_optimiser(args.optimiser, args.lr)

    # Optimise!
    params, _ = optimise_hyperparameters(
        params=params,
        kernel=kernel,
        optimiser=optimiser,
        data=training_set,
        num_epochs=args.num_epochs,
    )

    print("Optimised parameters:")
    print(f"  length scales: {jnp.exp(params['log_lengthscale'])}")
    print(f"  variance: {jnp.exp(params['log_variance'])}")

    # Use the optimised parameters:
    k = eq(
        lengthscale=jnp.exp(params["log_lengthscale"]),
        variance=jnp.exp(params["log_variance"]),
    )
    gp = gpr.fit(training_set.X, training_set.y, k)

    circle_mean, covar = gpr.predict(test_set.X, gp)
    circle_std = jnp.sqrt(jnp.diag(covar))

    grid_mean, covar = gpr.predict(function_data.X, gp)
    grid_std = jnp.sqrt(jnp.diag(covar))

    # Reshape to grid and compute gradients:
    grid_mean = grid_mean.reshape(args.resy, args.resx)
    grid_std = grid_std.reshape(args.resy, args.resx)

    cosine_similarity = compute_cosine_similarity(function_data.dy, grid_mean)

    # ==================== Plotting ====================
    _, ax = plt.subplots(
        3,
        2,
        figsize=(12, 9),
        layout="constrained",
    )

    grid_extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]
    vmin = np.min(jnp.array([function_data.y.min(), grid_mean.min()]))
    vmax = np.max(jnp.array([function_data.y.max(), grid_mean.max()]))

    true_c = ax[0, 0].imshow(
        function_data.y,
        extent=grid_extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax[0, 0].plot(*test_set.X.T, ls="--", marker="", color="w", alpha=0.5, zorder=5)
    ax[0, 0].scatter(
        *training_set.X.T,
        s=18,
        c=training_set.f.ravel(),
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
        edgecolors="w",
        linewidths=0.5,
        alpha=1,
        zorder=10,
    )

    ax[0, 1].quiver(
        *function_data.X.T,
        *function_data.dy.T,
        function_data.y.ravel(),
        angles="xy",
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    ax[0, 1].plot(*test_set.X.T, ls="--", marker="", color="k", alpha=0.5, zorder=5)
    ax[0, 1].quiver(
        *training_set.X.T,
        *training_set.y.T,
        training_set.f.ravel(),
        angles="xy",
        scale_units="xy",
        scale=2,
        width=0.005,
        norm=Normalize(vmin=vmin, vmax=vmax),
        zorder=10,
    )

    pred_c = ax[1, 0].imshow(
        grid_mean,
        extent=grid_extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1, 0].plot(*test_set.X.T, ls="--", marker="", color="w", alpha=0.5, zorder=5)
    ax[1, 0].scatter(
        *training_set.X.T,
        s=18,
        c=training_set.f.ravel(),
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
        edgecolors="w",
        linewidths=0.5,
        alpha=1,
        zorder=10,
    )

    diff_c = ax[1, 1].imshow(
        cosine_similarity,
        extent=grid_extent,
        origin="lower",
        cmap="Spectral",
        vmin=-1,
        vmax=1,
    )
    ax[1, 1].plot(*test_set.X.T, ls="--", marker="", color="w", alpha=0.5)

    ax[2, 0].plot(test_set.theta, test_set.f, c="C1", ls="--", label="True function")
    ax[2, 0].scatter(training_set.theta, training_set.f, c="C3", label="Training data")
    ax[2, 0].fill_between(
        test_set.theta,
        circle_mean.squeeze() - 2 * circle_std,
        circle_mean.squeeze() + 2 * circle_std,
        color="C0",
        alpha=0.3,
        label="2-sigma confidence",
    )
    ax[2, 0].plot(test_set.theta, circle_mean, c="C0", ls="-", label="Predicted mean")

    var_c = ax[2, 1].imshow(
        2 * grid_std,
        extent=grid_extent,
        origin="lower",
        cmap="Purples",
        vmin=0,
    )
    ax[2, 1].plot(*test_set.X.T, ls="--", marker="", color="w", alpha=0.5)

    ax[0, 1].set_xlim(x_array[0], x_array[-1])
    ax[0, 1].set_ylim(y_array[0], y_array[-1])
    ax[2, 0].set_xlim(0, 2 * jnp.pi)
    ax[0, 1].set_aspect("equal")
    ax[0, 0].set_aspect("equal")

    plt.colorbar(true_c, ax=ax[0, 0])
    plt.colorbar(pred_c, ax=ax[1, 0])
    plt.colorbar(diff_c, ax=ax[1, 1])
    plt.colorbar(var_c, ax=ax[2, 1])

    ax[0, 0].set_title("True potential")
    ax[0, 1].set_title("Gradient field")
    ax[1, 0].set_title("Predicted potential")
    ax[1, 1].set_title("Cosine similarity of gradients")
    ax[2, 0].set_title("Prediction along boundary")
    ax[2, 1].set_title("Predictive uncertainty on potential values (2 std)")

    ax[2, 0].set_xlabel(r"$\theta$/radians")
    ax[2, 0].set_ylabel("Potential value")

    # plt.show()
    plt.savefig("gp_on_boundary.pdf")


if __name__ == "__main__":
    main()
