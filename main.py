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


def cov_matrix(
    kernel: Callable, x: Float[Array, "N D"], y: Float[Array, "M D"]
) -> Float[Array, "N M"]:
    K = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(y))(x)
    return K


def eq(lengthscale: Float[Array, "D"], variance: float) -> Callable:
    "The exponentiated quadratic covariance function."

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
        return variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))

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

    def ensure_2d_function_values(
        f: Float[Array, "M"] | Float[Array, "M 1"],
    ) -> Float[Array, "M 1"]:
        "Make sure function values have shape [M 1]."
        return jax.lax.cond(
            len(f.shape) == 1, lambda x: jnp.atleast_2d(x).T, lambda x: x, f
        )

    def A(dx: Float[Array, "N D"], dy: Float[Array, "N D"]) -> Float[Array, "N N"]:
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

    def B(dx: Float[Array, "N D"], y: Float[Array, "M 1"]) -> Float[Array, "N M"]:
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

    def C(x: Float[Array, "M 1"], dy: Float[Array, "N D"]) -> Float[Array, "M N"]:
        batched_matrix = ddy_cov(x, dy)
        matrix = jnp.concatenate(batched_matrix, axis=1)
        return matrix

    # ===== Constructing the covariance function for D =====
    # This is just the covariance function itself.

    def D(x: Float[Array, "M 1"], y: Float[Array, "M 1"]) -> Float[Array, "M M"]:
        return cov_matrix(kernel, x, y)

    return CovMatrix(A, B, C, D)


def main() -> None:
    args = parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # Set up toy dataset:

    x_array = jnp.linspace(0, 10, args.resx)
    y_array = jnp.linspace(0, 5, args.resy)
    xx, yy, dfdx, dfdy, f = generate_toydata(subkey, x_array, y_array)

    # embed()
    # Define training set:
    train_frac = 0.03
    key, subkey = random.split(key)
    train_idx = random.choice(
        subkey,
        args.resx * args.resy,
        shape=(int(args.resx * args.resy * train_frac),),
        replace=False,
    )

    X = jnp.dstack((xx, yy)).reshape(-1, 2)
    df = jnp.dstack((dfdx, dfdy)).reshape(-1, 2)
    # f = f.reshape(-1, 1)

    Xtrain = X[train_idx]
    dftrain = df[train_idx]

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)

    Cov = derivative_cov_func(k)

    K = Cov.A(Xtrain, Xtrain)
    k = Cov.B(Xtrain, X)  # Xtrain, Xtest

    L = jax.scipy.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]), lower=True)
    alpha = jax.scipy.linalg.cho_solve((L, True), dftrain.reshape(-1, 1))

    f_pred = k.T @ alpha
    v = jax.scipy.linalg.solve_triangular(L, k, lower=True)
    var = Cov.D(X, X) - v.T @ v

    # f = f.reshape(args.resy, args.resx)
    f_pred = f_pred.reshape(args.resy, args.resx)
    # f_diff = f_pred - f

    dfdy_pred, dfdx_pred = jnp.gradient(f_pred)
    f_diff = jnp.sqrt((dfdx_pred - dfdx) ** 2 + (dfdy_pred - dfdy) ** 2)

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(1.2 * (x_array[-1] - x_array[0]), 1.2 * (y_array[-1] - y_array[0])),
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
        f_diff,
        extent=[x_array[0], x_array[-1], y_array[0], y_array[-1]],
        origin="upper",
        cmap="Spectral",
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

    ax[0, 0].set_title("True function")
    ax[0, 1].set_title("Function gradient")
    ax[1, 0].set_title("Predicted function")
    ax[1, 1].set_title("Gradient difference")

    # plt.show()
    plt.savefig("gp_on_derivative_obs_toy_data.pdf")


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
