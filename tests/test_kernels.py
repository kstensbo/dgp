from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Scalar

from dgp import kernels
from dgp.settings import _default_jitter

from IPython import embed  # noqa: F401

jax.config.update("jax_enable_x64", True)


def eq_deriv(
    lengthscale: Float[Array, "D"], variance: Scalar | float, i: int, j: int
) -> Callable:
    "Implements the ith and jth partial derivatives of the EQ kernel."

    dirac = 1.0 if i == j else 0.0

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Scalar:
        factor = (dirac - ((x[i] - y[i]) * (x[j] - y[j])) / (lengthscale**2)) / (
            lengthscale**2
        )
        return factor * variance * jnp.exp(-0.5 * jnp.sum(((x - y) / lengthscale) ** 2))

    return k


def diag_div_free_kernel(
    lengthscale: Float[Array, "D"], variance: Scalar | float
) -> Callable:
    "The special case of a diagonal divergence-free kernel based on the EQ kernel."

    k_eq = kernels.eq(lengthscale, variance)

    def k(x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "D D"]:
        # (x - y) / l: [D, 1]
        scaled_diff = ((x - y) / lengthscale)[:, None]
        factor = (2 - jnp.sum(scaled_diff**2)) * jnp.eye(
            len(x)
        ) + scaled_diff @ scaled_diff.T

        assert factor.shape == (len(x), len(x))

        return factor * k_eq(x, y) / (lengthscale**2)

    return k


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

    def test_zero_divergence(self) -> None:
        "Test that samples from the diagonal kernel have zero divergence."

        key = random.PRNGKey(seed=0)

        k = diag_div_free_kernel(self.lengthscale, 1.0)

        Nx = 10
        Ny = 10
        Nz = 4
        M = Nx * Ny * Nz

        # Set up toy dataset:
        x_array = jnp.linspace(0, 1, Nx)
        y_array = jnp.linspace(0, 1, Ny)
        z_array = jnp.linspace(0, 0.1 * Nz, Nz)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        assert X.shape == (M, 3)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )

        C = cov(X, X)
        assert C.shape == (M, M, 3, 3)

        K = kernels.tensor_to_matrix(C)
        assert K.shape == (3 * M, 3 * M)
        assert jnp.all(K[:3, :3] == C[0, 0])
        assert jnp.all(K[:3, 3:6] == C[0, 1])
        assert jnp.all(K[3:6, 0:3] == C[1, 0])
        assert jnp.all(K[3:6, 3:6] == C[1, 1])
        assert jnp.all(K[3:6, :3] == K[:3, 3:6])

        isnan = True
        scaling = 1
        while isnan:
            L = jnp.linalg.cholesky(K + scaling * _default_jitter * jnp.eye(K.shape[0]))
            isnan = jnp.any(jnp.isnan(L))
            if isnan:
                scaling *= 10

        assert not jnp.any(jnp.isnan(L))

        # Sample noise:
        u = random.normal(key, shape=[K.shape[0], 1])

        # Transform noise to GP prior samples of the vector field:
        f = L @ u

        assert f.shape == (3 * M, 1)

        f_grid = f.squeeze().reshape(M, 3)
        f_grid_3d = f_grid.reshape(Nx, Ny, Nz, 3)
        f3 = f_grid_3d

        # Function derivative:
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]
        dz = z_array[1] - z_array[0]

        dfdx = np.gradient(f3[..., 0], dx, axis=0, edge_order=2)  # [N, N, N]
        dfdy = np.gradient(f3[..., 1], dy, axis=1, edge_order=2)  # [N, N, N]
        dfdz = np.gradient(f3[..., 2], dz, axis=2, edge_order=2)  # [N, N, N]

        # df_grid = jnp.dstack((dfdx, dfdy, dfdz)).reshape(-1, 3)
        # div = dfdx + dfdy + dfdz
        # #
        # import matplotlib.pyplot as plt
        #
        # num_rows = int(np.ceil(np.sqrt(Nz)))
        # num_cols = int(np.ceil(np.sqrt(Nz)))
        #
        # fig, ax = plt.subplots(
        #     num_rows,
        #     num_cols,
        #     figsize=(12, 8),
        #     squeeze=False,
        # )
        # dx = (x_array[-1] - x_array[0]) / Nx / 2
        # dy = (y_array[-1] - y_array[0]) / Ny / 2
        #
        # for i in range(num_rows):
        #     for j in range(num_cols):
        #         if i * num_cols + j >= Nz:
        #             pass
        #         else:
        #             ax[i, j].imshow(
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
        #             ax[i, j].set_title(f"z={i * num_cols + j}")
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
            atol=dx**2 + dy**2 + dz**2,
        )


class TestDivFreeKernel:
    x = jnp.ones(3, dtype=float)
    y = jnp.zeros(3, dtype=float)
    lengthscale = jnp.ones(3, dtype=float)
    k_eq = staticmethod(kernels.eq(lengthscale, 1.0))

    def test_output_shape(self) -> None:
        k = kernels.div_free(self.k_eq)
        assert k(self.x, self.y).shape == (3, 3)

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
                expected_output += (
                    kernels.levi_civita(i, k, l)
                    * kernels.levi_civita(j, m, n)
                    * dk(self.x, self.y)
                )

        assert jnp.allclose(output, expected_output)
