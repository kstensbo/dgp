from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from dgp import kernels

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
        factor = (2 - jnp.sum(scaled_diff**2)) / (lengthscale**2) * jnp.eye(
            len(x)
        ) + scaled_diff @ scaled_diff.T / (lengthscale**2)

        return factor * k_eq(x, y)

    return k


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
