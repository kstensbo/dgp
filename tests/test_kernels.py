from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jaxtyping import Array, Float, Scalar
from matplotlib.colors import CenteredNorm, SymLogNorm

from dgp import kernels
from dgp.settings import _default_jitter

from IPython import embed  # noqa

jax.config.update("jax_enable_x64", True)


def eq_deriv(
    lengthscale: Float[Array, "D"], variance: Scalar | float, i: int, j: int
) -> Callable:
    "Implements the ith and jth partial derivatives of the EQ kernel."

    dirac = 1.0 if i == j else 0.0

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Scalar:
        # factor = (dirac - ((x[i] - y[i]) * (x[j] - y[j])) / (lengthscale**2)) / (
        #     lengthscale**2
        # )
        # factor = dirac / lengthscale[i] ** 2 - ((x[i] - y[i]) * (x[j] - y[j])) / (
        #     lengthscale[i] ** 2 * lengthscale[j] ** 2
        # )
        #
        scaled_diff = (x - y) / lengthscale**2
        factor = dirac / lengthscale[i] ** 2 - (scaled_diff[i] * scaled_diff[j])

        return factor * variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))

    return k


def diag_div_free_kernel(
    lengthscale: Float[Array, "D"], variance: Scalar | float
) -> Callable:
    "The special case of a diagonal divergence-free kernel based on the EQ kernel."

    k_eq = kernels.eq(lengthscale, variance)
    dimension = len(lengthscale)

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "D D"]:
        # (x - y) / l: [D, 1]
        scaled_diff = ((x - y) / lengthscale)[:, None]
        factor = (dimension - 1 - jnp.sum(scaled_diff**2)) * jnp.eye(
            len(x)
        ) + scaled_diff @ scaled_diff.T

        factor /= lengthscale**2

        # assert factor.shape == (len(x), len(x))

        return factor * k_eq(x, y)

    return k


def field_and_divergence(
    grid: Float[Array, "N N 2"] | Float[Array, "N N N 3"], p: int, q: int
) -> tuple[
    Float[Array, "N N 2"] | Float[Array, "N N N 3"],
    Float[Array, "N N"] | Float[Array, "N N N"],
]:
    # Define field:
    # norm = jnp.sum(grid**2, axis=-1) ** (p / 2)
    # f = grid / norm[..., None]
    # div = (grid.shape[-1] - p) / norm
    #
    # q = 1
    norm = (jnp.sum(grid**2, axis=-1) + q) ** (p / 2)
    f = grid / norm[..., None]
    # div = (grid.shape[-1] - p) / norm + (
    #     # p * q * jnp.sum(grid**2, axis=-1) ** ((p - 2) / 2) / norm**2
    #     p * q / (jnp.sum(grid**2, axis=-1) ** ((p + 2) / 2))
    # )
    div = (grid.shape[-1] - p + p * q / (jnp.sum(grid**2, axis=-1) + q)) / norm

    return f, div


def compute_curl_from_potential(
    grid: Float[Array, "Nx Ny"] | Float[Array, "Nx Ny Nz"], stepsizes: Float[Array, "D"]
) -> Float[Array, "Nx Ny"] | Float[Array, "Nx Ny Nz"]:
    df = jnp.gradient(grid, *stepsizes)

    dfydx = jnp.gradient(df[1], stepsizes[1], axis=0)
    dfxdy = jnp.gradient(df[0], stepsizes[0], axis=1)

    curl_z = dfydx - dfxdy
    curl = curl_z

    if len(grid.shape) == 3:  # noqa: PLR2004
        dfxdz = jnp.gradient(df[0], stepsizes[0], axis=2)
        dfydz = jnp.gradient(df[1], stepsizes[1], axis=2)

        dfzdx = jnp.gradient(df[2], stepsizes[2], axis=0)
        dfzdy = jnp.gradient(df[2], stepsizes[2], axis=1)

        curl_x = dfzdy - dfydz
        curl_y = -(dfzdx - dfxdz)
        curl = jnp.stack((curl_x, curl_y, curl_z), axis=-1)

    return curl


class TestEQDerivative:
    x = jnp.ones(3, dtype=float)
    y = jnp.zeros(3, dtype=float)
    lengthscale = jnp.ones(3, dtype=float)

    # To avoid automatically transforming the kernel into an instance method, we wrap it
    # in staticmethod:
    k = staticmethod(kernels.eq(lengthscale, 1.0))

    def test_eq(self) -> None:
        # Sanity check:
        assert self.k(self.x, self.x) == 1.0

    def test_equal_inputs_and_equal_partial(self) -> None:
        # With i = j and x = y, the output should be k_eq(x, y)/lengthscale**2:
        dk = eq_deriv(self.lengthscale, 1.0, 0, 0)
        output = dk(self.x, self.x)

        expected_output = self.k(self.x, self.x) / (self.lengthscale**2)

        assert jnp.allclose(output, expected_output)

    def test_equal_inputs_and_not_equal_partial(self) -> None:
        # With i != j and x = y, the output should be 0:
        dk = eq_deriv(self.lengthscale, 1.0, 0, 1)
        output = dk(self.x, self.x)

        expected_output = 0.0

        assert jnp.allclose(output, expected_output)

    def test_not_equal_inputs_and_equal_partial(self) -> None:
        # With i = j and x != y, the output should be:
        dk = eq_deriv(self.lengthscale, 1.0, 0, 0)
        output = dk(self.x, self.y)

        expected_output = (
            1 / (self.lengthscale**2) - 1 / (self.lengthscale**4)
        ) * self.k(self.x, self.y)

        assert jnp.allclose(output, expected_output)

    def test_not_equal_inputs_and_not_equal_partial(self) -> None:
        # With i != j and x != y, the output should be:
        dk = eq_deriv(self.lengthscale, 1.0, 0, 1)
        output = dk(self.x, self.y)

        expected_output = (-1 / (self.lengthscale**4)) * self.k(self.x, self.y)

        assert jnp.allclose(output, expected_output)


class TestMultiOutputKernel:
    "Various tests to verify transformations to multi-output covariance matrices."

    def test_construction(self) -> None:
        "Test construction of [D*N, D*N] matrix from [N, N, D, D] tensor."
        N = 2
        D = 3
        inner_cov = np.arange(D * D).reshape(D, D)
        full_cov = np.tile(inner_cov, (N, N, 1, 1))

        assert full_cov.shape == (N, N, D, D)

        tiled_cov = np.transpose(full_cov, (0, 2, 1, 3))
        # tiled_cov = np.transpose(full_cov, (2, 3, 0, 1))
        # tiled_cov = np.transpose(full_cov.reshape((N, N, D * D)), (2, 0, 1))
        tiled_cov = tiled_cov.reshape((D * N, D * N))
        # print(tiled_cov)

        assert np.all(tiled_cov[:D, :D] == inner_cov)
        assert np.all(tiled_cov[D : 2 * D, :D] == inner_cov)
        assert np.all(tiled_cov[:D, D : 2 * D] == inner_cov)
        assert np.all(tiled_cov[D : 2 * D, D : 2 * D] == inner_cov)

    def test_convenience_function(self) -> None:
        """
        Test convenience function for converting [N, N, D, D] tensor to a tiled
        [D*N, D*N] matrix.
        """

        N = 2
        D = 3
        inner_cov = np.arange(D * D).reshape(D, D)
        full_cov = np.tile(inner_cov, (N, N, 1, 1))

        tiled_cov = np.transpose(full_cov, (0, 2, 1, 3))
        expected_output = tiled_cov.reshape((D * N, D * N))

        output = kernels.tensor_to_matrix(jnp.array(full_cov, dtype=float))
        assert np.all(output == expected_output)


class TestGridConstruction:
    def test_grid_construction(self) -> None:
        "Test that we understand the structure of the constructed grid."
        N = 5

        x_array = np.arange(N)
        y_array = np.arange(N)
        z_array = np.arange(N)
        xx, yy, zz = np.meshgrid(x_array, y_array, z_array, indexing="ij")

        grid = np.stack([xx, yy, zz], axis=-1)
        X = grid.reshape(-1, 3)
        X2 = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        assert np.allclose(X, X2)

        assert np.all(X[:N, 0] == np.zeros(N))
        assert np.all(X[:N, 1] == np.zeros(N))
        assert np.all(X[:N, 2] == np.arange(N))

        # The following assumes `indexing="xy"`:
        # assert np.all(X[N : 2 * N, 0] == np.ones(N))
        # assert np.all(X[N : 2 * N, 1] == np.zeros(N))
        # assert np.all(X[N : 2 * N, 2] == np.arange(N))

        # The following assumes `indexing="ij"`:
        assert np.all(X[N : 2 * N, 0] == np.zeros(N))
        assert np.all(X[N : 2 * N, 1] == np.ones(N))
        assert np.all(X[N : 2 * N, 2] == np.arange(N))

    def test_order_of_sampled_function_compared_to_the_original_X(self) -> None:
        """
        Test that going from a covariance tensor of shape (M, M, D, D) to a (D*M, D*M)
        matrix preserves the order of the original list of grid points, such that the
        grid can be reconstructed correctly.
        """

        def f(x: Float[Array, "D 1"], y: Float[Array, "D 1"]) -> Float[Array, "D D"]:
            """
            Toy covariance function which results in the overall multi-output covariance
            function being the flattened grid coordinates both at the top row and the
            left column.
            """

            # Repeat x along columns, such that x = [x1, x2, x3] will be one column.
            Kx = jnp.stack([x, x, x], axis=1)

            # Repeat y along rows, such that y = [y1, y2, y3] will be one row.
            Ky = jnp.stack([y, y, y], axis=0)

            K = Kx + Ky

            return K

        N = 10
        M = N**3
        # Set up toy dataset:
        x_array = jnp.arange(N)
        y_array = jnp.arange(N)
        z_array = jnp.arange(N)

        xx, yy, zz = np.meshgrid(x_array, y_array, z_array)

        # Create input coordinate grid
        X = jnp.stack(
            [xx.flatten(), yy.flatten(), zz.flatten()], axis=1
        )  # Shape (M, 3)

        # Compute covariance matrix
        vf = jax.vmap(
            jax.vmap(f, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1
        )
        K = vf(X, X)  # Shape (M, M, 3, 3)

        assert K.shape == (M, M, 3, 3)

        # Flatten the covariance matrix.
        K_flat = kernels.tensor_to_matrix(K.astype(float))

        # Pick the first column of K in a way that resembles the sampling. The first
        # column should contain the coordinates in a flattened list.
        pick_first_column = jnp.zeros((3 * M, 1), dtype=float)
        pick_first_column = pick_first_column.at[0].set(1.0)
        f_sampled = K_flat @ pick_first_column

        # Reconstruct the grid
        f_grid = f_sampled.reshape(M, 3)  # Shape (M, 3)

        assert np.allclose(f_grid, X)

        f_grid_3d = f_grid.reshape((N, N, N, 3))
        assert np.allclose(f_grid_3d, jnp.stack([xx, yy, zz], axis=-1))

    def test_reconstruction_from_convenience_function(self) -> None:
        """
        Test that noise sampled from a GP with multi-output covariance matrix
        constructed from the convenience function can be transformed back to a grid
        correctly.
        """

        N = 5
        M = N**3

        # The multi-output covariance matrix is constructed by tiling the matrix-valued
        # covariances for each input. This following simulates the vector
        #   vec = [x_111, y_111, z_111, x_112, y_112, z_112, ..., x_nnn, y_nnn, z_nnn]
        # which need to be reconstructed as
        #   v[1,1,1] = [x_111, y_111, z_111],
        #   v[1,1,2] = [x_112, y_112, z_112],
        #   ...
        #   v[1,1,n] = [x_11n, y_11n, z_11n],
        #   v[1,2,1] = [x_121, y_121, z_121],
        #   ...
        #   v[n,n,n] = [x_nnn, y_nnn, z_nnn]
        #
        vec = np.concatenate([np.arange(3, dtype=int) + n for n in range(M)], axis=0)
        assert vec.shape == (3 * M,)

        # v1 = vec.reshape(-1, 3)
        # v1 = np.stack([vec[d * M : (d + 1) * M] for d in range(3)], axis=1).squeeze()
        v1 = np.stack([vec[3 * m : 3 * (m + 1)] for m in range(M)], axis=0).squeeze()
        assert v1.shape == (M, 3)

        v2 = v1.reshape(N, N, N, 3)
        # embed()
        assert np.all(v2[0, 0, 0] == [0, 1, 2])
        assert np.all(v2[0, 0, 1] == [1, 2, 3])
        assert np.all(v2[0, 0, 2] == [2, 3, 4])

        # v2[0, 1, 0]: one loop through the last axis.
        assert np.all(v2[0, 1, 0] == np.arange(3) + N)

        # v2[1, 0, 0]: one loop through the second axis, for which each index is a full
        # loop through the last axis.
        assert np.all(v2[1, 0, 0] == np.arange(3) + N * N)

        # v2[1, 1, 0]: one loop through the fist axis plus one through the second.
        assert np.all(v2[1, 1, 0] == np.arange(3) + N * N + N)

        assert np.all(v2[3, 2, 1] == np.arange(3) + 3 * N**2 + 2 * N + 1)


class TestCovarianceMatrix:
    def test_cov_matrix_computation(self) -> None:
        "Test that the covariance matrix is computed correctly, e.g., not transposed."

        def k(x: Float[Array, "N D"], y: Float[Array, "M D"]) -> Float[Array, "N M"]:
            return jnp.sum(x + y)

        X = jnp.arange(5).reshape(-1, 1)
        Y = jnp.arange(10, 20).reshape(-1, 1)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )

        K = cov(X, Y)

        assert K.shape == (5, 10)
        assert jnp.all(K[0] == jnp.arange(10, 20))
        assert jnp.all(K[1] == jnp.arange(10, 20) + 1)

    def test_multioutput_cov_matrix_transformation(self) -> None:
        """
        Test that the covariance matrix is computed correctly for the case of
        matrix-valued kernels.
        """

        def k(x: Float[Array, "N D"], y: Float[Array, "M D"]) -> Float[Array, "N M"]:
            return jnp.arange(9).reshape(3, 3) + jnp.sum(x + y)

        X = jnp.arange(5).reshape(-1, 1)
        Y = jnp.arange(10, 20).reshape(-1, 1)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        K = cov(X, Y)

        assert K.shape == (5, 10, 3, 3)
        assert jnp.all(K[0, 0] == jnp.arange(9).reshape(3, 3) + 0 + 10)
        assert jnp.all(K[1, 0] == jnp.arange(9).reshape(3, 3) + 1 + 10)
        assert jnp.all(K[2, 2] == jnp.arange(9).reshape(3, 3) + 2 + 12)


class TestCurlFreeKernel:
    def test_zero_curl(self) -> None:
        """
        Test that a potential sampled from the EQ covariance function lead to curl-free
        field.
        """

        # Construct 3D grid:
        Nx = 8
        Ny = 8
        Nz = 8
        N = Nx * Ny * Nz

        stepsizes = jnp.array([0.1, 0.1, 0.1])

        x_array = jnp.linspace(0, Nx * stepsizes[0], Nx)
        y_array = jnp.linspace(0, Ny * stepsizes[1], Ny)
        z_array = jnp.linspace(0, Nz * stepsizes[2], Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        k = kernels.eq(lengthscale=jnp.ones(3, dtype=float), variance=1.0)

        K = kernels.cov_matrix(k, X, X)
        L = jnp.linalg.cholesky(K + _default_jitter * jnp.eye(N))

        # Sample noise:
        key = random.PRNGKey(seed=0)
        u = random.normal(key, shape=[N, 1])

        # Transform noise to GP prior sample:
        f = L @ u
        f_grid = f.reshape(Nx, Ny, Nz)

        # Potentials sampled from the curl-free kernel should be curl-free:
        curl = compute_curl_from_potential(f_grid, stepsizes)

        assert jnp.allclose(jnp.zeros_like(curl), curl)


class TestDiagDivFreeKernel:
    x = jnp.ones(3, dtype=float)
    y = jnp.zeros(3, dtype=float)
    lengthscale = jnp.ones(3, dtype=float)
    k_eq = staticmethod(kernels.eq(lengthscale, 1.0))

    def test_output_shape(self) -> None:
        k = diag_div_free_kernel(self.lengthscale, 1.0)
        assert k(self.x, self.y).shape == (3, 3)

    def test_outer_product(self) -> None:
        scaled_diff = jnp.atleast_2d((self.x - self.y) / self.lengthscale).T
        assert jnp.allclose(scaled_diff @ scaled_diff.T, jnp.ones((3, 3), dtype=float))

    def test_simple_input(self) -> None:
        term1 = (2 - 3) * jnp.eye(3)
        term2 = jnp.ones((3, 3), dtype=float)

        expected_output = (term1 + term2) * self.k_eq(self.x, self.y)

        k = diag_div_free_kernel(self.lengthscale, 1.0)
        output = k(self.x, self.y)

        assert jnp.allclose(output, expected_output)

    def test_vector_notation(self) -> None:
        """
        Test that the vector notation version of the kernel gives the same result as the
        scalar notation version.
        """

        def scalar_diag_div_free_kernel(
            x: Float[Array, "D"], y: Float[Array, "D"], i: int, j: int
        ) -> float:
            delta = 1 if i == j else 0

            lm2 = 1 / self.lengthscale[0] ** 2

            factor = (
                lm2
                * (
                    2 * delta
                    - lm2 * delta * (jnp.sum((x - y) ** 2))
                    + lm2 * (x[i] - y[i]) * (x[j] - y[j])
                )
            ).squeeze()

            return factor * self.k_eq(x, y)

        k = diag_div_free_kernel(self.lengthscale, 1.0)
        output = k(self.x, self.y)

        for i in range(3):
            for j in range(3):
                expected = scalar_diag_div_free_kernel(self.x, self.y, i, j)

                assert output[i, j] == expected

    def test_grid_reconstruction(self) -> None:
        "Test that the grid can be flattened and reconstructed correctly."
        # Set up toy dataset:
        x_array = jnp.linspace(0, 1, 10)
        y_array = jnp.linspace(0, 1, 10)
        z_array = jnp.linspace(0, 1, 10)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array)

        assert xx.shape == (10, 10, 10)

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        f = X.reshape((10, 10, 10, 3))

        assert jnp.all(f[..., 0] == xx)
        assert jnp.all(f[..., 1] == yy)
        assert jnp.all(f[..., 2] == zz)

    def test_analytical_divergence_2d(self) -> None:
        plot = False

        Nx = 128
        Ny = 128

        # Set up toy dataset:
        x_array = jnp.linspace(-1, 1, Nx)
        y_array = jnp.linspace(-1, 1, Ny)

        xx, yy = jnp.meshgrid(x_array, y_array, indexing="ij")

        # Construct input coordinates for grid, shape [M, 2]:
        grid = jnp.stack([xx, yy], axis=-1)

        f, div = field_and_divergence(grid, p=2, q=1)

        # Function derivative:
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]

        dfdx = np.gradient(f[..., 0], dx, axis=0)  # [N, N]
        dfdy = np.gradient(f[..., 1], dy, axis=1)  # [N, N]

        empirical_div = dfdx + dfdy

        # ======================== Visual inspection =============================
        if plot:
            _, ax = plt.subplots(2, 3, layout="constrained")
            dx = (x_array[-1] - x_array[0]) / Nx / 2
            dy = (y_array[-1] - y_array[0]) / Ny / 2

            c_fx = ax[0, 0].imshow(
                # Flipping x and y for plotting purposes.
                f[..., 1],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_fy = ax[1, 0].imshow(
                # Flipping x and y for plotting purposes.
                f[..., 0],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_dfxdx = ax[0, 1].imshow(
                # Flipping x and y for plotting purposes.
                dfdy,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_dfydy = ax[1, 1].imshow(
                # Flipping x and y for plotting purposes.
                dfdx,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_div = ax[0, 2].imshow(
                div,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
            )
            ax[0, 2].quiver(
                yy,
                xx,
                f[..., 1],
                f[..., 0],
                angles="xy",
                scale_units="xy",
                # scale=20,
            )
            c_emdiv = ax[1, 2].imshow(
                empirical_div,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                # norm="log",
            )
            ax[0, 0].set_title("$f_x$")
            ax[1, 0].set_title("$f_y$")
            ax[0, 1].set_title("$\\partial f_x / \\partial x$")
            ax[1, 1].set_title("$\\partial f_y / \\partial y$")
            ax[0, 2].set_title("$\\nabla \\cdot f$ (analytical)")
            ax[1, 2].set_title("$\\nabla \\cdot f$ (empirical)")
            plt.colorbar(c_fx)
            plt.colorbar(c_fy)
            plt.colorbar(c_dfxdx)
            plt.colorbar(c_dfydy)
            plt.colorbar(c_div)
            plt.colorbar(c_emdiv)
            plt.show()
        # ========================================================================

        assert jnp.allclose(div[1:-1, 1:-1], empirical_div[1:-1, 1:-1], atol=0.1)

    def test_analytical_divergence_3d(self) -> None:
        plot = False

        Nx = 32
        Ny = 32
        Nz = 32

        # Set up toy dataset:
        x_array = jnp.linspace(-1, 1, Nx)
        y_array = jnp.linspace(-1, 1, Ny)
        z_array = jnp.linspace(-1, 1, Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [M, 2]:
        grid = jnp.stack([xx, yy, zz], axis=-1)

        # NOTE: q != 0 leads to numerical instabilities close to x=0.
        f, div = field_and_divergence(grid, p=3, q=1)

        # Function derivative:
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]
        dz = z_array[1] - z_array[0]

        dfdx = np.gradient(f[..., 0], dx, axis=0)  # [N, N]
        dfdy = np.gradient(f[..., 1], dy, axis=1)  # [N, N]
        dfdz = np.gradient(f[..., 2], dz, axis=2)  # [N, N]

        empirical_div = dfdx + dfdy + dfdz

        # ======================== Visual inspection =============================
        if plot:
            _, ax = plt.subplots(2, 2, layout="constrained")
            dx = (x_array[-1] - x_array[0]) / Nx / 2
            dy = (y_array[-1] - y_array[0]) / Ny / 2
            dz = (z_array[-1] - z_array[0]) / Nz / 2

            slice_id = Nz // 2

            # c_fx = ax[0, 0].imshow(
            #     # Flipping x and y for plotting purposes.
            #     f[..., slice_id, 1],
            #     extent=[
            #         x_array[0] - dx,
            #         x_array[-1] + dx,
            #         y_array[0] - dy,
            #         y_array[-1] + dy,
            #     ],
            #     origin="lower",
            #     cmap="coolwarm",
            #     norm=CenteredNorm(),
            # )
            # c_fy = ax[1, 0].imshow(
            #     # Flipping x and y for plotting purposes.
            #     f[..., slice_id, 0],
            #     extent=[
            #         x_array[0] - dx,
            #         x_array[-1] + dx,
            #         y_array[0] - dy,
            #         y_array[-1] + dy,
            #     ],
            #     origin="lower",
            #     cmap="coolwarm",
            #     norm=CenteredNorm(),
            # )
            # c_dfxdx = ax[0, 1].imshow(
            #     # Flipping x and y for plotting purposes.
            #     dfdy[..., slice_id],
            #     extent=[
            #         x_array[0] - dx,
            #         x_array[-1] + dx,
            #         y_array[0] - dy,
            #         y_array[-1] + dy,
            #     ],
            #     origin="lower",
            #     cmap="coolwarm",
            #     norm=CenteredNorm(),
            # )
            # c_dfydy = ax[1, 1].imshow(
            #     # Flipping x and y for plotting purposes.
            #     dfdx[..., slice_id],
            #     extent=[
            #         x_array[0] - dx,
            #         x_array[-1] + dx,
            #         y_array[0] - dy,
            #         y_array[-1] + dy,
            #     ],
            #     origin="lower",
            #     cmap="coolwarm",
            #     norm=CenteredNorm(),
            # )
            c_div = ax[0, 0].imshow(
                div[..., slice_id],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
            )
            ax[0, 0].quiver(
                yy[..., slice_id],
                xx[..., slice_id],
                f[..., slice_id, 1],
                f[..., slice_id, 0],
                angles="xy",
                scale_units="xy",
                # scale=40,
            )
            c_emdiv = ax[1, 0].imshow(
                empirical_div[..., slice_id],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                norm=CenteredNorm(),
            )
            res = empirical_div[..., slice_id] - div[..., slice_id]
            c_res = ax[0, 1].imshow(
                res,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                norm=SymLogNorm(
                    1e-3, vmin=-np.max(np.abs(res)), vmax=np.max(np.abs(res))
                ),
                cmap="coolwarm",
            )
            # ax[0, 0].set_title("$f_x$")
            # ax[1, 0].set_title("$f_y$")
            # ax[0, 1].set_title("$\\partial f_x / \\partial x$")
            # ax[1, 1].set_title("$\\partial f_y / \\partial y$")
            ax[0, 0].set_title("$\\nabla \\cdot f$ (analytical)")
            ax[1, 0].set_title("$\\nabla \\cdot f$ (empirical)")
            ax[0, 1].set_title("Residual (empirical - analytical)")
            # plt.colorbar(c_fx)
            # plt.colorbar(c_fy)
            # plt.colorbar(c_dfxdx)
            # plt.colorbar(c_dfydy)
            plt.colorbar(c_div)
            plt.colorbar(c_emdiv)
            plt.colorbar(c_res)
            plt.show()
        # ========================================================================

        assert jnp.allclose(
            div[1:-1, 1:-1, 1:-1], empirical_div[1:-1, 1:-1, 1:-1], atol=0.1
        )

    def test_zero_divergence_2d(self) -> None:  # noqa: PLR0915
        "Test that samples from the diagonal kernel have zero divergence."

        plot = False

        key = random.PRNGKey(seed=0)

        k = diag_div_free_kernel(jnp.ones(2, dtype=float), 1.0)

        Nx = 64
        Ny = 64
        M = Nx * Ny

        # Set up toy dataset:
        x_array = jnp.linspace(0, 1, Nx)
        y_array = jnp.linspace(0, 1, Ny)

        xx, yy = jnp.meshgrid(x_array, y_array, indexing="ij")

        # Construct input coordinates for grid, shape [M, 2]:
        X = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        assert X.shape == (M, 2)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )

        C = cov(X, X)
        assert C.shape == (M, M, 2, 2)

        K = kernels.tensor_to_matrix(C)

        isnan = True
        scaling = 1e-6
        while isnan:
            L = jnp.linalg.cholesky(K + scaling * _default_jitter * jnp.eye(K.shape[0]))
            isnan = jnp.any(jnp.isnan(L))
            if isnan:
                scaling *= 10

        # print(
        #     f"Added 1e{int(np.log10(scaling * _default_jitter))} to the diagonal "
        #     "for the Cholesky decomposition."
        # )

        assert not jnp.any(jnp.isnan(L))

        # Sample noise:
        u = random.normal(key, shape=[K.shape[0], 1])

        # Transform noise to GP prior samples of the vector field:
        f = L @ u

        assert f.shape == (2 * M, 1)

        f_grid = f.squeeze().reshape(M, 2)
        f_grid_2d = f_grid.reshape(Nx, Ny, 2)
        f2d = f_grid_2d

        # Computing the derivative of the field using the derivative kernel:
        # ddx = jax.jacfwd(k, argnums=0)
        # ddx_cov = jax.vmap(
        #     jax.vmap(ddx, in_axes=(0, None), out_axes=0),
        #     in_axes=(None, 0),
        #     out_axes=2,
        # )
        # deriv_cov = kernels.derivative_cov_func(k)

        # # This derivative function outputs a [D, D] matrix for a pair of inputs:
        # d2kdxdy = jax.jacfwd(jax.jacrev(k, argnums=0), argnums=1)
        #
        # # vmap over each input to the covariance function in turn. We want each
        # # argument to keep their mutual order in the resulting matrix, hence the
        # # out_axes specifications:
        # d2kdxdy_cov = jax.vmap(
        #     jax.vmap(d2kdxdy, in_axes=(0, None), out_axes=0),
        #     in_axes=(None, 0),
        #     out_axes=1,
        # )
        #
        # # def ensure_2d_function_values(
        # #     f: Float[Array, "M"] | Float[Array, "M 1"],
        # # ) -> Float[Array, "M 1"]:
        # #     "Make sure function values have shape [M 1]."
        # #     return jax.lax.cond(
        # #         len(f.shape) == 1, lambda x: jnp.atleast_2d(x).T, lambda x: x, f
        # #     )
        #
        # def A(
        #     dx: Float[Array, "N D"], dy: Float[Array, "N D"]
        # ) -> Float[Array, "D*D*N D*D*N D D"]:
        #     batched_matrix = d2kdxdy_cov(dx, dy)
        #     matrix = jnp.concatenate(jnp.concatenate(batched_matrix, axis=1), axis=1)
        #     return matrix
        #
        # C_df = A(X, X)
        # K_df = kernels.tensor_to_matrix(C_df)
        #
        # # ddx_cov = jax.vmap(
        # #     jax.vmap(ddx, in_axes=(0, None), out_axes=0),
        # #     in_axes=(None, 0),
        # #     out_axes=1,
        # # )
        # # C_df2 = ddx_cov2(X, X)
        #
        # # C_df = ddx_cov(X, X)
        # # C_df_stacked = jnp.concatenate(C_df, axis=1)
        # # K_df = kernels.tensor_to_matrix(C_df_stacked)
        #
        # isnan = True
        # scaling = 1e-6
        # while isnan:
        #     L_df = jnp.linalg.cholesky(
        #         K_df + scaling * _default_jitter * jnp.eye(K_df.shape[0])
        #     )
        #     isnan = jnp.any(jnp.isnan(L_df))
        #     if isnan:
        #         scaling *= 10
        #
        # print(
        #     f"Added 1e{int(np.log10(scaling * _default_jitter))} to the diagonal "
        #     "for the Cholesky decomposition."
        # )
        #
        # assert not jnp.any(jnp.isnan(L_df))
        #
        # # Sample noise:
        # u = random.normal(key, shape=[K_df.shape[0], 1])
        #
        # # Transform noise to GP prior samples of the vector field:
        # dF = L_df @ u
        #
        # assert dF.shape == (2 * 2 * M, 1)
        #
        # # Undo tensor_to_matrix:
        # dF_points = dF.squeeze().reshape(2 * M, 2)
        #
        # # Undo concatenation:
        # dF_points_dxdy = jnp.stack(jnp.split(dF_points, 2), axis=-1)  # [M, D, P]
        # dF_2d = dF_points_dxdy.reshape(Nx, Ny, 2, 2)
        #
        # # Compute divergence as
        # div_k = jnp.diagonal(
        #     dF_2d,
        #     axis1=2,
        #     axis2=3,
        # ).sum(-1)
        # # embed()

        # Function derivative:
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]

        dfdx = np.gradient(f2d[..., 0], dx, axis=0)  # [N, N]
        dfdy = np.gradient(f2d[..., 1], dy, axis=1)  # [N, N]

        # ======================== Visual inspection =============================
        if plot:
            div = dfdx + dfdy

            _, ax = plt.subplots(2, 3, layout="constrained")
            dx = (x_array[-1] - x_array[0]) / Nx / 2
            dy = (y_array[-1] - y_array[0]) / Ny / 2

            c_fx = ax[0, 0].imshow(
                # Flipping x and y for plotting purposes.
                f2d[..., 1],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_fy = ax[1, 0].imshow(
                # Flipping x and y for plotting purposes.
                f2d[..., 0],
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_dfxdx = ax[0, 1].imshow(
                # Flipping x and y for plotting purposes.
                dfdy,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_dfydy = ax[1, 1].imshow(
                # Flipping x and y for plotting purposes.
                dfdx,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            c_div = ax[0, 2].imshow(
                div,
                extent=[
                    x_array[0] - dx,
                    x_array[-1] + dx,
                    y_array[0] - dy,
                    y_array[-1] + dy,
                ],
                origin="lower",
                cmap="coolwarm",
                norm=CenteredNorm(),
            )
            ax[0, 2].quiver(
                yy,
                xx,
                f2d[..., 1],
                f2d[..., 0],
                angles="xy",
                scale_units="xy",
                # scale=20,
            )
            # c_div_k = ax[1, 2].imshow(
            #     div_k,
            #     extent=[
            #         x_array[0] - dx,
            #         x_array[-1] + dx,
            #         y_array[0] - dy,
            #         y_array[-1] + dy,
            #     ],
            #     origin="lower",
            # )
            ax[0, 0].set_title("$f_x$")
            ax[1, 0].set_title("$f_y$")
            ax[0, 1].set_title("$\\partial f_x / \\partial x$")
            ax[1, 1].set_title("$\\partial f_y / \\partial y$")
            ax[0, 2].set_title("$\\nabla \\cdot f$ (empirical)")
            # ax[1, 2].set_title("$\\nabla \\cdot f$ (kernel)")
            plt.colorbar(c_fx)
            plt.colorbar(c_fy)
            plt.colorbar(c_dfxdx)
            plt.colorbar(c_dfydy)
            plt.colorbar(c_div)
            # plt.colorbar(c_div_k)
            plt.show()
        # ========================================================================

        divergence = dfdx + dfdy

        # For comparison to zero, ignore the box edges as the errors will be much larger
        # here.
        # embed()
        inner_divergence = jnp.ravel(divergence[1:-1, 1:-1])

        # The array of zeros should be the first argument to make the relative tolerance
        # actually do something.
        # The value of atol matches the expected error:
        # https://numpy.org/doc/2.0/reference/generated/numpy.gradient.html
        assert jnp.allclose(
            jnp.zeros_like(inner_divergence),
            inner_divergence,
            # atol=dx**2 + dy**2,
            atol=0.1,
        )

    def test_zero_divergence_3d(self) -> None:
        "Test that samples from the diagonal kernel have zero divergence."

        key = random.PRNGKey(seed=0)

        k = diag_div_free_kernel(self.lengthscale, 1.0)

        Nx = 16
        Ny = 16
        Nz = 4
        M = Nx * Ny * Nz

        D = 3

        # Set up toy dataset:
        x_array = jnp.linspace(0, 0.1, Nx)
        y_array = jnp.linspace(0, 0.1, Ny)
        z_array = jnp.linspace(0, 0.01 * Nz, Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [M, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, D)
        assert X.shape == (M, D)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )

        C = cov(X, X)
        assert C.shape == (M, M, D, D)

        K = kernels.tensor_to_matrix(C)
        assert K.shape == (D * M, D * M)
        assert jnp.all(K[:3, :3] == C[0, 0])
        assert jnp.all(K[:3, 3:6] == C[0, 1])
        assert jnp.all(K[3:6, 0:3] == C[1, 0])
        assert jnp.all(K[3:6, 3:6] == C[1, 1])
        assert jnp.all(K[3:6, :3] == K[:3, 3:6])

        isnan = True
        scaling = 1e-6
        while isnan:
            L = jnp.linalg.cholesky(K + scaling * _default_jitter * jnp.eye(K.shape[0]))
            isnan = jnp.any(jnp.isnan(L))
            if isnan:
                scaling *= 10

        # print(
        #     f"Added 1e{int(np.log10(scaling * _default_jitter))} to the diagonal "
        #     "for the Cholesky decomposition."
        # )
        assert not jnp.any(jnp.isnan(L))

        # Sample noise:
        u = random.normal(key, shape=[K.shape[0], 1])

        # Transform noise to GP prior samples of the vector field:
        f = L @ u

        assert f.shape == (D * M, 1)

        f_grid = f.squeeze().reshape(M, D)
        f_grid_3d = f_grid.reshape(Nx, Ny, Nz, D)
        f3 = f_grid_3d

        # Function derivative:
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]
        dz = z_array[1] - z_array[0]

        dfdx = np.gradient(f3[..., 0], dx, axis=0, edge_order=2)  # [N, N, N]
        dfdy = np.gradient(f3[..., 1], dy, axis=1, edge_order=2)  # [N, N, N]
        dfdz = np.gradient(f3[..., 2], dz, axis=2, edge_order=2)  # [N, N, N]

        # df_grid = jnp.dstack((dfdx, dfdy, dfdz)).reshape(-1, D)
        # div = dfdx + dfdy + dfdz
        #

        # num_rows = int(np.ceil(np.sqrt(Nz)))
        # num_cols = int(np.ceil(np.sqrt(Nz)))

        # fig, ax = plt.subplots(
        #     num_rows,
        #     num_cols,
        #     figsize=(12, 8),
        #     squeeze=False,
        #     layout="constrained",
        # )
        # dx = (x_array[-1] - x_array[0]) / Nx / 2
        # dy = (y_array[-1] - y_array[0]) / Ny / 2
        #
        # for i in range(num_rows):
        #     for j in range(num_cols):
        #         if i * num_cols + j >= Nz:
        #             pass
        #         else:
        #             cax = ax[i, j].imshow(
        #                 div[..., i * num_cols + j],
        #                 extent=[
        #                     x_array[0] - dx,
        #                     x_array[-1] + dx,
        #                     y_array[0] - dy,
        #                     y_array[-1] + dy,
        #                 ],
        #                 origin="lower",
        #             )
        #             ax[i, j].quiver(
        #                 xx[..., 0],
        #                 yy[..., 0],
        #                 f3[..., i * num_cols + j, 0],
        #                 f3[..., i * num_cols + j, 1],
        #                 angles="xy",
        #             )
        #             ax[i, j].set_title(f"z={z_array[i * num_cols + j]:3.2g}")
        #
        #             plt.colorbar(cax)
        #
        # plt.show()

        # divergence = jnp.ravel(dfdx + dfdy + dfdz)
        divergence = dfdx + dfdy + dfdz

        # For comparison to zero, ignore the box edges as the errors will be much larger
        # here.
        # embed()
        inner_divergence = jnp.ravel(divergence[1:-1, 1:-1, 1:-1])

        # The array of zeros should be the first argument to make the relative tolerance
        # actually do something.
        # The value of atol matches the expected error:
        # https://numpy.org/doc/2.0/reference/generated/numpy.gradient.html
        assert jnp.allclose(
            jnp.zeros_like(inner_divergence),
            inner_divergence,
            # atol=dx**2 + dy**2 + dz**2,
            atol=0.1,
        )


class TestDivFreeKernel:
    x = jnp.ones(3, dtype=float)
    y = jnp.zeros(3, dtype=float)
    lengthscale = jnp.ones(3, dtype=float)
    k_eq = staticmethod(kernels.eq(lengthscale, 1.0))

    def test_output_shape(self) -> None:
        k = kernels.div_free(self.k_eq)
        assert k(self.x, self.y).shape == (3, 3)

    def test_diagonal_special_case(self) -> None:
        """
        Compare a diagonal divergence-free kernel computed using autodiff to the
        version from Wahlström et al. (2013).
        """

        Nx = 8
        Ny = 8
        Nz = 8

        # Set up toy dataset:
        x_array = jnp.linspace(-1, 1, Nx)
        y_array = jnp.linspace(-1, 1, Ny)
        z_array = jnp.linspace(-1, 1, Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [M, 2]:
        grid = jnp.stack([xx, yy, zz], axis=-1)

        # This field is divergence-free, so it should be modelled well by the kernels.
        f, _ = field_and_divergence(grid, p=3, q=0)

        X = grid.reshape(-1, 3)
        f_flat = f.reshape(-1, 3)

        # Split the data in random training and test sets.
        key = random.PRNGKey(0)
        shuffle_idx = jax.random.choice(key, len(X), shape=(len(X),), replace=False)
        X_train, X_test = jax.numpy.split(X[shuffle_idx], [16])
        f_train, _ = jax.numpy.split(f_flat[shuffle_idx], [16])

        predictions = []
        covariances = []
        Ks = []

        k_diag = diag_div_free_kernel(self.lengthscale, 1.0)
        # k_df = kernels.div_free(self.k_eq)

        # TODO: create diagonal divergence-free kernel.
        def k_zero(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
            return 0.0

        base_kernels = [
            [self.k_eq, k_zero, k_zero],
            [k_zero, self.k_eq, k_zero],
            [k_zero, k_zero, self.k_eq],
        ]
        k_df_diag = kernels.div_free(base_kernels)
        # embed()

        for k_df in [k_df_diag, k_diag]:
            C = kernels.cov_matrix(k_df, X_train, X_train)
            K = kernels.tensor_to_matrix(C)
            Ks.append(K)

            L = jax.scipy.linalg.cholesky(
                K + _default_jitter * jnp.eye(K.shape[0]), lower=True
            )
            alpha = jax.scipy.linalg.cho_solve((L, True), f_train.reshape(-1, 1))

            C_pred = kernels.cov_matrix(k_df, X_train, X_test)
            K_pred = kernels.tensor_to_matrix(C_pred)
            f_pred = K_pred.T @ alpha
            v = jax.scipy.linalg.solve_triangular(L, K_pred, lower=True)
            C_pp = kernels.cov_matrix(k_df, X_test, X_test)
            K_pp = kernels.tensor_to_matrix(C_pp)
            covar = K_pp - v.T @ v

            predictions.append(f_pred)
            covariances.append(covar)

        assert jnp.allclose(predictions[0], predictions[1])
        assert jnp.allclose(covariances[0], covariances[1])

    def test_for_eq_base_kernel(self) -> None:
        # For the EQ kernel, the output can be computed using the derivative defined
        # above.

        k_df = kernels.div_free(self.k_eq)
        output = k_df(self.x, self.y)

        # List of non-zero Levi-Civita symbols:
        nonzero_levi_civita_symbols = [
            (1, 2, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1),
            (1, 3, 2),
            (2, 1, 3),
        ]

        expected_output = jnp.zeros((3, 3), dtype=float)
        for i, k, l in nonzero_levi_civita_symbols:  # noqa: E741
            for j, m, n in nonzero_levi_civita_symbols:
                dk = eq_deriv(self.lengthscale, 1.0, k, m)
                expected_output = expected_output.at[i - 1, j - 1].add(
                    kernels.levi_civita(i, k, l)
                    * kernels.levi_civita(j, m, n)
                    * dk(self.x, self.y)
                )

        assert jnp.allclose(output, expected_output)


class TestCurlAndDivFreeKernel:
    def test_zero_curl(self) -> None:
        """
        Test that a potential sampled from the curl- and divergence-free covariance
        function lead to a curl-free field.
        """

        # Construct 3D grid:
        Nx = 8
        Ny = 8
        Nz = 8
        N = Nx * Ny * Nz

        stepsizes = jnp.array([0.1, 0.1, 0.1])

        x_array = jnp.linspace(0, Nx * stepsizes[0], Nx)
        y_array = jnp.linspace(0, Ny * stepsizes[1], Ny)
        z_array = jnp.linspace(0, Nz * stepsizes[2], Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # # First sample a potential and compute a field.
        # k_eq = kernels.eq(lengthscale=jnp.ones(3), variance=1.0)
        #
        # K = kernels.cov_matrix(k_eq, X, X)
        # L = jnp.linalg.cholesky(K + _default_jitter * jnp.eye(N))
        #
        # # Sample noise:
        # key = random.PRNGKey(seed=0)
        # key, subkey = random.split(key)
        # u = random.normal(subkey, shape=[N, 1])
        #
        # # Transform noise to GP prior sample:
        # f = L @ u
        # f_grid = f.reshape(Nx, Ny, Nz)
        #
        # # Compute field:
        # field = jnp.gradient(f_grid, *stepsizes)
        #
        # # Choose a small, random subset of the field vectors:
        # key, subkey = random.split(key)
        # ids = random.choice(subkey, N, shape=(8,))
        #
        # X_train = X[ids]
        # field_train = field.reshape(-1, 3)[ids]
        #
        # Model the field using the curl- and divergence-free kernel:

        base_kernel = kernels.eq(lengthscale=jnp.ones(3, dtype=float), variance=1.0)
        k = kernels.cdf_kernel(base_kernel=base_kernel, i=0, j=0)

        # K = kernels.cov_matrix(k, X_train, X_train)
        K = kernels.cov_matrix(k, X, X)

        assert K.shape == (N, N)

        L = jnp.linalg.cholesky(K + _default_jitter * jnp.eye(N))

        # Sample noise:
        key = random.PRNGKey(seed=0)
        key, subkey = random.split(key)
        u = random.normal(subkey, shape=[N, 1])

        # Transform noise to GP prior sample:
        f = L @ u
        f_grid = f.reshape(Nx, Ny, Nz)

        # Potentials sampled from the curl-free kernel should be curl-free:
        curl = compute_curl_from_potential(f_grid, stepsizes)

        assert jnp.allclose(jnp.zeros_like(curl), curl)

    def test_zero_divergence(self) -> None:
        """
        Test that a potential sampled from the curl- and divergence-free covariance
        function lead to a divergence-free field.
        """
        # Construct 3D grid:
        Nx = 8
        Ny = 8
        Nz = 8
        N = Nx * Ny * Nz

        # Take small step sizes to avoid large numerical errors in the divergence:
        stepsizes = np.array(
            [
                0.1,
                0.1,
                0.1,
            ]
        )

        x_array = jnp.linspace(0, Nx * stepsizes[0], Nx)
        y_array = jnp.linspace(0, Ny * stepsizes[1], Ny)
        z_array = jnp.linspace(0, Nz * stepsizes[2], Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        k_eq = kernels.eq(lengthscale=jnp.ones(3, dtype=float), variance=1.0)
        base_kernel = [[k_eq] * 3] * 3

        k = jax.jit(kernels.cdf_kernel(base_kernel=base_kernel, i=0, j=0))

        # import time
        # s = time.time()
        # k = jax.jit(kernels.cdf_kernel(base_kernel=k_eq, i=0, j=0)) # Fast
        # k = jax.jit(kernels.cdf_kernel(base_kernel=base_kernel, i=0, j=0)) # Slow
        # print()
        # print(f"Setting up the kernel: {time.time() - s:.3g} s")
        #
        # s = time.time()
        # K = kernels.cov_matrix(k, X, X)
        # print(f"Computing covariance matrix: {time.time() - s:.3g} s")

        K = kernels.cov_matrix(k, X, X)
        assert K.shape == (N, N)

        jitter = 1e-12
        L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))

        # Sample noise:
        key = random.PRNGKey(seed=0)
        key, subkey = random.split(key)
        u = random.normal(subkey, shape=[N, 1])

        # Transform noise to GP prior sample:
        f = L @ u
        f_grid = f.reshape(Nx, Ny, Nz)

        field = np.gradient(f_grid, *stepsizes)

        dfdx = np.gradient(field[0], stepsizes[0], axis=0)  # [N, N]
        dfdy = np.gradient(field[1], stepsizes[1], axis=1)  # [N, N]
        dfdz = np.gradient(field[2], stepsizes[2], axis=2)  # [N, N]

        # df = np.gradient(np.stack(field, axis=-1), *stepsizes, axis=-1)

        empirical_div = dfdx + dfdy + dfdz

        # The edges are poorly estimated, and this propagates when taking the second
        # gradient, hence we cut the edges twice:
        inner_div = empirical_div[1:-1, 1:-1, 1:-1]
        inner_div = inner_div[1:-1, 1:-1, 1:-1]

        assert jnp.allclose(jnp.zeros_like(inner_div), inner_div, atol=0.1)
