import jax.numpy as jnp

def identity(dim: int) -> jnp.ndarray:
    """

    """
    return jnp.identity(dim, dtype=jnp.complex64)

def pauli_x() -> jnp.ndarray:
    """

    """
    return jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)