from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Scalar

from dgp import kernels
from dgp.settings import _default_jitter

from IPython import embed  # noqa: F401


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

        return factor * k_eq(x, y) / (lengthscale**2)

    return k


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
        N = 2
        D = 3
        inner_cov = np.arange(D * D).reshape(D, D)
        full_cov = np.tile(inner_cov, (N, N, 1, 1))

        tiled_cov = np.transpose(full_cov, (0, 2, 1, 3))
        expected_output = tiled_cov.reshape((D * N, D * N))

        output = kernels.tensor_to_matrix(jnp.array(full_cov, dtype=float))
        assert np.all(output == expected_output)


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
        "Test that samples from the diagonal kernel have zero divergence"

        key = random.PRNGKey(seed=0)

        k = diag_div_free_kernel(self.lengthscale, 1.0)

        # Set up toy dataset:
        x_array = jnp.linspace(0, 1, 10)
        y_array = jnp.linspace(0, 1, 10)
        z_array = jnp.linspace(0, 1, 10)

        xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array)

        # Construct input coordinates for grid, shape [N, 3]:
        X = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # vmap over each input to the covariance function in turn. We want each argument
        # to keep their mutual order in the resulting matrix, hence the out_axes
        # specifications:
        cov = jax.vmap(
            jax.vmap(k, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        K = kernels.tensor_to_matrix(cov(X, X))
        assert K.shape == (3 * 1000, 3 * 1000)

        isnan = True
        scaling = 1
        while isnan:
            L = jnp.linalg.cholesky(K + scaling * _default_jitter * jnp.eye(K.shape[0]))
            isnan = jnp.any(jnp.isnan(L))
            if isnan:
                scaling *= 10

        # print(scaling)
        assert not jnp.any(jnp.isnan(L))

        # Sample noise:
        u = random.normal(key, shape=[K.shape[0], 1])

        # Transform noise to GP prior sample:
        f = L @ u

        assert f.shape == (3 * 1000, 1)

        f = f.reshape(-1, 3)
        assert f.shape == (1000, 3)
        f = f.reshape(x_array.shape[0], y_array.shape[0], z_array.shape[0], 3)

        # Function derivative:
        # dfdx, dfdy, dfdz = jnp.gradient(f_grid)
        dx = x_array[1] - x_array[0]
        dy = y_array[1] - y_array[0]
        dz = z_array[1] - z_array[0]

        dfdx = jnp.gradient(f[..., 0], dx, axis=0)  # [N, N, N]
        dfdy = jnp.gradient(f[..., 1], dy, axis=1)  # [N, N, N]
        dfdz = jnp.gradient(f[..., 2], dz, axis=2)  # [N, N, N]

        # df_grid = jnp.dstack((dfdx, dfdy, dfdz)).reshape(-1, 3)
        divergence = jnp.ravel(dfdx + dfdy + dfdz)

        assert jnp.allclose(divergence, jnp.zeros_like(divergence))


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
