import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float

import dgp.regression as gpr
from dgp.kernels import cov_matrix, eq

from IPython import embed  # noqa

_jitter = 1e-6

# Set JAX to use 64bit:
jax.config.update("jax_enable_x64", True)


def generate_toydata(
    key: ArrayLike,
    x_array: ArrayLike,
    y_array: ArrayLike,
) -> tuple[
    Float[Array, "Ny Nx"],
    Float[Array, "Ny Nx"],
    Float[Array, "Ny Nx"],
    Float[Array, "Ny Nx"],
    Float[Array, "Ny Nx"],
]:
    xx, yy = jnp.meshgrid(x_array, y_array)

    X = jnp.stack([xx, yy], axis=-1).reshape(x_array.shape[0] * y_array.shape[0], 2)

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
    K = cov_matrix(k, X, X)
    L = jnp.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]))

    u = random.normal(key, shape=[X.shape[0], 1])

    f = L @ u

    # Derivative of y:
    f_grid = f.reshape(y_array.shape[0], x_array.shape[0])
    f_grid = jnp.flipud(f_grid)
    dfdy, dfdx = jnp.gradient(f_grid)

    return xx, yy, dfdx, dfdy, f_grid


def main() -> None:
    args = parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # Set up toy dataset:
    x_array = jnp.linspace(0, 10, args.resx)
    y_array = jnp.linspace(0, 5, args.resy)

    xx, yy, dfdx, dfdy, f = generate_toydata(subkey, x_array, y_array)

    X = jnp.dstack((xx, yy)).reshape(-1, 2)
    df = jnp.dstack((dfdx, dfdy)).reshape(-1, 2)

    # Define training set:
    train_frac = 0.03
    key, subkey = random.split(key)
    train_idx = random.choice(
        subkey,
        args.resx * args.resy,
        shape=(int(args.resx * args.resy * train_frac),),
        replace=False,
    )

    # Attempt at using training observations from a circle:
    # c = jnp.array([5.0, 2.5])
    # r = 1.5
    # theta = jnp.linspace(0, 2 * jnp.pi, 20)
    # xc = r * jnp.cos(theta) + c[0]
    # yc = r * jnp.sin(theta) + c[1]
    #
    # Xb = jnp.dstack((xc, yc)).squeeze()
    #
    # # f = f.reshape(-1, 1)
    #
    # idx = []
    # for i in range(Xb.shape[0]):
    #     min_idx = jnp.argmin(jnp.sum((X - Xb[i]) ** 2, axis=1))
    #     idx.append(min_idx)
    #
    # train_idx = jnp.array(idx)

    Xtrain = X[train_idx]
    dftrain = df[train_idx]

    # ==================== Optimise the GP ====================

    # Define the covariance function to use and the initial hyperparameters:
    kernel = eq
    params = gpr.tune(kernel, Xtrain, dftrain, num_epochs=args.num_epochs)

    print("Optimised parameters:")
    for key, val in params.items():
        print(f"  {key} = {val}")

    # Use the optimised parameters:
    k = eq(**params)
    gp = gpr.fit(Xtrain, dftrain, k)

    f_pred, covar = gpr.predict(X, gp)
    std = jnp.sqrt(jnp.diag(covar))

    # Reshape to grid and compute gradients:
    f_pred = f_pred.reshape(args.resy, args.resx)
    std = std.reshape(args.resy, args.resx)

    dfdy_pred, dfdx_pred = jnp.gradient(f_pred)
    df_pred = jnp.dstack((dfdx_pred, dfdy_pred)).reshape(-1, 2)

    # Cosine similarity between predicted and true gradients:
    cosine_similarity = jnp.sum(df * df_pred, axis=1) / jnp.sqrt(
        jnp.sum(df**2, axis=1) * jnp.sum(df_pred**2, axis=1)
    )
    cosine_similarity = cosine_similarity.reshape(args.resy, args.resx)

    # ==================== Plotting ====================
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(1.2 * (x_array[-1] - x_array[0]), 1.8 * (y_array[-1] - y_array[0])),
        layout="constrained",
    )
    true_c = ax[0, 0].imshow(
        f,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
    )
    ax[0, 0].scatter(*Xtrain.T, s=12, facecolor=plt.cm.Oranges(0.5), alpha=1)

    ax[0, 1].quiver(xx, yy, dfdx.ravel(), dfdy.ravel(), f.ravel(), angles="xy")

    pred_c = ax[1, 0].imshow(
        f_pred,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
    )
    ax[1, 0].scatter(*Xtrain.T, s=12, facecolor=plt.cm.Oranges(0.5), alpha=1)

    diff_c = ax[1, 1].imshow(
        # f_diff,
        cosine_similarity,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
        cmap="Spectral",
        # cmap="Purples",
        vmin=-1,
        vmax=1,
    )

    ax[2, 0].plot(
        logger["epoch"], logger["logp"], c="C0", label="Marginal log-likelihood"
    )

    var_c = ax[2, 1].imshow(
        std,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
        cmap="Purples",
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
    ax[2, 1].set_title("Predictive uncertainty (1 std)")

    ax[2, 0].set_xlabel("Epoch")
    ax[2, 0].set_ylabel("Marginal log-likelihood")

    plt.show()
    # plt.savefig("gp_on_derivative_obs_toy_data.pdf")


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
        "--num_epochs", type=int, default=1000, help="Number of training epochs."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
