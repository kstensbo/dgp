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


def test_div_free() -> None:
    # Define test points
    x = jnp.ones(3, dtype=float)
    y = jnp.zeros(3, dtype=float)

    # Use an exponentiated quadratic covariance function as base kernel:
    k_eq = kernels.eq(jnp.ones(3, dtype=float), 1.0)

    # Construct divergence-free covariance function:
    k_df = kernels.div_free(k_eq)

    # Compute covariance between function values:
    kxy = k_df(x, y)

    assert kxy.shape == (3, 3)
