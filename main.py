import argparse
from collections.abc import Callable
from typing import NamedTuple

# import cola
# import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython import embed  # noqa
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from dgp.kernels import eq, cov_matrix
import dgp.regression as gpr

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

    # kernel = gpx.kernels.RBF(
    #    lengthscale=jnp.array([1.0, 1.0]), variance=jnp.array([1.0])
    # )
    # K = kernel.gram(X)
    ## L = jnp.linalg.cholesky(K)
    # L = cola.Cholesky()(K + _jitter * cola.ops.I_like(K))

    # K2 = K.to_dense()
    # L2 = jnp.linalg.cholesky(K2 + _jitter * jnp.eye(K2.shape[0]))

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
    K = cov_matrix(k, X, X)
    L = jnp.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]))

    u = random.normal(key, shape=[X.shape[0], 1])

    f = L @ u
    # y2 = L2 @ u

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

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)

    gp = gpr.fit(Xtrain, dftrain, k)

    f_pred, std = gpr.predict(X, gp)

    f_pred = f_pred.reshape(args.resy, args.resx)
    std = std.reshape(args.resy, args.resx)
    # f_diff = f_pred - f

    dfdy_pred, dfdx_pred = jnp.gradient(f_pred)
    df_pred = jnp.dstack((dfdx_pred, dfdy_pred)).reshape(-1, 2)
    # f_diff = jnp.sqrt((dfdx_pred - dfdx) ** 2 + (dfdy_pred - dfdy) ** 2)

    # Cosine similarity between gradients:
    cosine_similarity = jnp.sum(df * df_pred, axis=1) / jnp.sqrt(
        jnp.sum(df**2, axis=1) * jnp.sum(df_pred**2, axis=1)
    )
    cosine_similarity = cosine_similarity.reshape(args.resy, args.resx)

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

    ax[0, 1].quiver(xx, yy, dfdx.ravel(), dfdy.ravel(), f.ravel())

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

    var_c = ax[2, 1].imshow(
        std,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
        cmap="Purples",
    )

    ax[0, 1].set_xlim(x_array[0], x_array[-1])
    ax[0, 1].set_ylim(y_array[0], y_array[-1])
    # ax[1].set_ylim(y_array[-1], y_array[0])
    ax[0, 1].set_aspect("equal")
    # ax[1].set_aspect("equal", adjustable="datalim")
    # ax[1].autoscale()

    plt.colorbar(true_c, ax=ax[0, 0])
    plt.colorbar(pred_c, ax=ax[1, 0])
    plt.colorbar(diff_c, ax=ax[1, 1])
    plt.colorbar(var_c, ax=ax[2, 1])

    ax[0, 0].set_title("True function")
    ax[0, 1].set_title("Function gradient")
    ax[1, 0].set_title("Predicted function")
    ax[1, 1].set_title("Cosine similarity of gradients")
    ax[2, 1].set_title("Predictive uncertainty (1 std)")

    plt.show()
    # plt.savefig("gp_on_derivative_obs_toy_data.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--resx", type=int, default=50, help="Surface resolution in x direction."
    )
    parser.add_argument(
        "--resy", type=int, default=25, help="Surface resolution in y direction."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
