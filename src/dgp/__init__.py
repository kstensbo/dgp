from jaxtyping import install_import_hook

with install_import_hook("dgp", "beartype.beartype"):
    from dgp import kernels, regression

__all__ = [
    "kernels",
    "regression",
]

# import dgp.kernels
# import dgp.regression
