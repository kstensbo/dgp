import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from matplotlib import colors

from dgp import _default_jitter
from dgp.kernels import cov_matrix, eq

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

    args = parser.parse_args()
    return args


def generate_toydata(
    key: ArrayLike,
    x_array: ArrayLike,
    y_array: ArrayLike,
    num_training_data: int,
    _jitter: float = _default_jitter,
) -> tuple[
    Dataset,
    CircleDataset,
]:
    xx, yy = jnp.meshgrid(x_array, y_array)

    # Construct input coordinates for grid, shape [N, 2]:
    X_grid = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)

    # Sample training data:
    key, *subkey = random.split(key, 3)

    # For circle data:
    centre = jnp.array([x_array.mean(), y_array.mean()])
    radius = 1.5
    theta = random.uniform(
        subkey[0], shape=(num_training_data, 1), minval=0, maxval=2 * jnp.pi
    ).sort()
    theta = jnp.linspace(0, 2 * jnp.pi, num_training_data)[:, None]
    points_x = radius * jnp.cos(theta) + centre[0]
    points_y = radius * jnp.sin(theta) + centre[1]

    X_train = jnp.concatenate([points_x, points_y], axis=1)

    # # For a uniform distribution:
    # X_train = random.uniform(subkey[1], shape=(num_training_data, 2))
    # X_train = X_train.at[:, 0].set(
    #     X_train[:, 0] * (x_array[-1] - x_array[0]) + x_array[0]
    # )
    # X_train = X_train.at[:, 1].set(
    #     X_train[:, 1] * (y_array[-1] - y_array[0]) + y_array[0]
    # )

    # Additional locations for computing gradients at training locations:
    gradient_epsilon = 1e-2

    grad_x = jnp.linspace(-gradient_epsilon, gradient_epsilon, 3)
    grad_y = jnp.linspace(-gradient_epsilon, gradient_epsilon, 3)
    grad_mesh = jnp.array(jnp.meshgrid(grad_x, grad_y))
    X_train_mesh = grad_mesh + X_train[..., None, None]
    X_grad = jnp.rollaxis(X_train_mesh, 1, 4).reshape(-1, 2)
    # embed()

    def get_cross_locations(
        X: Float[Array, "N D"], epsilon: float = 1e-2
    ) -> Float[Array, "4 N D"]:
        dx = jnp.zeros_like(X).at[:, 0].set(epsilon)
        dy = jnp.zeros_like(X).at[:, 1].set(epsilon)
        locations = jnp.array(
            [
                X + dx,
                X - dx,
                X + dy,
                X - dy,
            ]
        )
        return locations

    # grad_loc = get_cross_locations(X_train, gradient_epsilon)
    # X_grad = jnp.concatenate(
    #     grad_loc,
    #     axis=0,
    # )

    # Additional locations for computing the vector field around the training locations
    # used to compute the curl.
    # In 2D, we need to compute dFy/dx - dFx/dy. For, e.g., dFy/dx, we must compute (Fx,
    # Fy) at both x + dx and x - dx

    # def get_cross_locations_for_curl(
    #     X: Float[Array, "N D"], epsilon: float = 1e-2
    # ) -> Float[Array, "12 N D"]:
    #     dx = jnp.zeros_like(X).at[:, 0].set(epsilon)
    #     dy = jnp.zeros_like(X).at[:, 1].set(epsilon)
    #
    #     # [4 N D]
    #     cross = get_cross_locations(X, epsilon)
    #
    #     locations = jnp.array(
    #         [
    #             # For X + dx:
    #             cross[0],
    #             cross[0] + dx,
    #             cross[0] + dy,
    #             # cross[0] - dy,
    #             # For X - dx:
    #             cross[1],
    #             cross[1] - dx,
    #             cross[1] + dy,
    #             # cross[1] - dy,
    #             # For X + dy:
    #             cross[2],
    #             cross[2] + dy,
    #             # cross[2] + dx,
    #             cross[2] - dx,
    #             # For X - dy:
    #             cross[3],
    #             cross[3] - dy,
    #             cross[3] + dx,
    #             cross[3] - dx,
    #         ]
    #     )
    #     return locations

    # Shift all training points:
    # X_train_shift = X_train + gradient_epsilon * jnp.ones_like(X_train)
    # X_grad_shift = jnp.concatenate(
    #     get_cross_locations(X_train_shift, gradient_epsilon), axis=0
    # )

    X = jnp.concatenate(
        (
            X_grid,
            X_train,
            X_grad,
            # X_train_shift,
            # X_grad_shift,
        ),
        axis=0,
    )

    k = eq(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
    K = cov_matrix(k, X, X)
    L = jnp.linalg.cholesky(K + _jitter * jnp.eye(K.shape[0]))

    # Sample noise:
    u = random.normal(key, shape=[X.shape[0], 1])

    # Transform noise to GP prior sample:
    f = L @ u

    (
        f_grid,
        f_train,
        f_grad,
        # f_train_shift,
        # f_grad_shift,
    ) = jnp.split(
        f,
        jnp.array(
            [
                X_grid.shape[0],
                X_train.shape[0],
                # X_grad.shape[0],
                # X_train_shift.shape[0],
            ]
        ).cumsum(),
    )

    f_grid = f_grid.reshape(y_array.shape[0], x_array.shape[0])
    # f_grid = jnp.flipud(f_grid)

    # Function derivative:
    dfdy, dfdx = jnp.gradient(f_grid)
    df_grid = jnp.dstack((dfdx, dfdy)).reshape(-1, 2)

    # Function derivative at training locations:
    # f_grad_dx_plus, f_grad_dx_minus, f_grad_dy_plus, f_grad_dy_minus = jnp.split(
    #     f_grad,
    #     jnp.array([num_training_data] * 3).cumsum(),
    # )
    f_grad = f_grad.reshape(-1, 3, 3)
    df_train_dy, df_train_dx = jnp.gradient(
        f_grad, gradient_epsilon, gradient_epsilon, axis=(1, 2)
    )

    df_train = jnp.dstack((df_train_dx[:, 1, 1], df_train_dy[:, 1, 1])).reshape(-1, 2)
    # f_grad = jnp.split(f_grad, 4)
    # df_train = jnp.concatenate(
    #     (
    #         (f_grad[0] - f_grad[1]) / (2 * gradient_epsilon),
    #         (f_grad[2] - f_grad[3]) / (2 * gradient_epsilon),
    #     ),
    #     # (
    #     #     (f_grad_dx_plus - f_grad_dx_minus) / (2 * gradient_epsilon),
    #     #     (f_grad_dy_plus - f_grad_dy_minus) / (2 * gradient_epsilon),
    #     # ),
    #     axis=1,
    # )
    assert df_train.shape == (num_training_data, 2)

    # f_grad_shift = jnp.split(f_grad_shift, 4)
    # df_train_shift = jnp.concatenate(
    #     (
    #         (f_grad_shift[0] - f_grad_shift[1]) / (2 * gradient_epsilon),
    #         (f_grad_shift[2] - f_grad_shift[3]) / (2 * gradient_epsilon),
    #     )
    # )

    function_data = Dataset(
        X=X_grid,
        y=f_grid,
        dy=df_grid,
        resx=x_array[1] - x_array[0],
        resy=y_array[1] - y_array[0],
    )

    training_set = CircleDataset(
        X=X_train,
        f=f_train,
        y=df_train,
        # y=jnp.zeros_like(X_train),
        theta=theta,
    )

    # training_set_shift = CircleDataset(
    #     X=X_train_shift,
    #     f=f_train_shift,
    #     y=df_train_shift,
    #     # y=jnp.zeros_like(X_train),
    #     theta=theta,
    # )

    return function_data, training_set  # , training_set_shift


def compute_curl_from_grid(data: Dataset) -> jnp.ndarray:
    f = data.y

    fy, fx = jnp.gradient(f, data.resy, data.resx)

    dfydx = jnp.gradient(fy, data.resx, axis=1)
    dfxdy = jnp.gradient(fx, data.resy, axis=0)

    curl = dfydx - dfxdy
    return curl


def main() -> None:
    args = parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # Set up toy dataset:
    x_array = jnp.linspace(0, 10, args.resx)
    y_array = jnp.linspace(0, 5, args.resy)

    (
        function_data,
        training_set,
        # training_set_shift,
    ) = generate_toydata(
        subkey, x_array, y_array, num_training_data=args.num_training_data
    )

    # Computing the curl:
    # curl_z = jnp.array()
    curl = compute_curl_from_grid(function_data)

    # ==================== Plotting ====================
    _, ax = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        layout="constrained",
        squeeze=False,
    )

    grid_extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]

    true_c = ax[0, 0].imshow(
        function_data.y,
        extent=grid_extent,
        origin="lower",
    )

    # ax[0, 0].scatter(*training_set.X.T, s=12, facecolor=plt.cm.Oranges(0.5), alpha=1)
    ax[0, 0].scatter(
        *training_set.X.T,
        s=15,
        c=training_set.f.ravel(),
        # facecolor=plt.cm.viridis(training_set.f.ravel()),
        cmap=plt.cm.viridis,
        vmin=function_data.y.min(),
        vmax=function_data.y.max(),
        edgecolors="w",
        linewidths=0.5,
        alpha=1,
    )

    ax[0, 1].quiver(
        *function_data.X.T,
        *function_data.dy.T,
        function_data.y.ravel(),
        angles="xy",
    )
    ax[0, 1].quiver(
        *training_set.X.T,
        *training_set.y.T,
        training_set.f.ravel(),
        angles="xy",
        scale_units="xy",
        scale=2,
        width=0.005,
        norm=colors.Normalize(vmin=function_data.y.min(), vmax=function_data.y.max()),
        # cmap=plt.cm.Reds,
    )

    div_c = ax[1, 0].imshow(
        function_data.dy.sum(axis=-1).reshape(function_data.y.shape),
        extent=grid_extent,
        origin="lower",
        norm=colors.CenteredNorm(),
        cmap="coolwarm",
    )

    curl_c = ax[1, 1].imshow(
        curl,
        extent=grid_extent,
        origin="lower",
        norm=colors.CenteredNorm(),
        cmap="coolwarm",
    )

    ax[0, 1].set_xlim(x_array[0], x_array[-1])
    ax[0, 1].set_ylim(y_array[0], y_array[-1])
    # ax[0, 1].set_ylim(y_array[-1], y_array[0])
    ax[0, 1].set_aspect("equal")
    ax[0, 0].set_aspect("equal")

    # ax[2, 0].set_ylim(bottom=-200, top=200)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    # fmt.set_powerlimits((0, 0))
    plt.colorbar(true_c, ax=ax[0, 0])
    plt.colorbar(div_c, ax=ax[1, 0], format=fmt)
    plt.colorbar(curl_c, ax=ax[1, 1], format=fmt)

    ax[0, 0].set_title("True function")
    ax[0, 1].set_title("Gradient field")

    ax[1, 0].set_title("Divergence")
    ax[1, 1].set_title("Curl")

    # plt.show()
    plt.savefig("toy_field.pdf")


if __name__ == "__main__":
    main()
