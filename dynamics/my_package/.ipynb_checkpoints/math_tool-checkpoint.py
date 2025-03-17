import jax
import jax.numpy as jnp

from jax import jit
from functools import partial

@partial(jit, static_argnums=(0, 1, ))
def tensor_prod(mat_1: jnp.ndarray, mat_2: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Kronecker product of two matrices A and B using JAX.
    
    Args:
        A (jnp.ndarray): First input matrix of shape (m, n).
        B (jnp.ndarray): Second input matrix of shape (p, q).
    
    Returns:
        jnp.ndarray: Kronecker product of A and B with shape (m*p, n*q).
    """
    return jnp.kron(mat_1, mat_2)

def normalize(state) -> jnp.ndarray:
    """
    Normalize a state vector or a density matrix.
    
    Args:
        state (jnp.array): Input state, either a vector (1D) or a density matrix (2D).
    
    Returns:
        jnp.array: Normalized state vector or density matrix.
    
    Raises:
        ValueError: If the input dimensions are not 1 (vector) or 2 (matrix),
                    or if the norm or trace is zero.
    """
    dim = jnp.ndim(state)
    
    if dim == 1:  # State vector normalization
        norm = jnp.linalg.norm(state)
        if norm == 0:
            raise ValueError("State vector has zero norm and cannot be normalized.")
        return state / norm
    
    elif dim == 2:  # Density matrix normalization
        trace = jnp.trace(state)
        if trace == 0:
            raise ValueError("Density matrix has zero trace and cannot be normalized.")
        return state / trace
    
    else:
        raise ValueError(f"Input dimensions do not match the expected dimensions. "
                         f"Got {dim}, but expected 1 (vector) or 2 (matrix).")

def factorial(n: int) -> int:
    """
    Compute the factorial of a natural number.
    Args:
        n (int): input a natural number.
    
    Returns:
        int: factorial of a natural number
    """
    def loop_func(i, acc):
        return acc*i
        
    return jax.lax.fori_loop(1, n+1, loop_func, 1)

def vec2dm(vec: jnp.ndarray) -> jnp.ndarray:
    """

    """
    return jnp.dot(vec, jax.lax.transpose(vec, (1, 0)))

def commutator(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """

    """
    return jnp.dot(A, B) - jnp.dot(B, A)

def anti_commutator(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """

    """
    return jnp.dot(A, B) + jnp.dot(B, A)

def dagger(A: jnp.ndarray) -> jnp.ndarray:
    """
    Return Hermitian transpose matrix
    Args:
        A (jnp.ndarray): input matrix of shape (m, n).

    Returns:
        jnp.ndarray: Hermitian transpose matrix
    """
    return jax.lax.conj(jax.lax.transpose(A, (1, 0)))

def kraus_repr(rho: jnp.ndarray, kraus_opr: jnp.ndarray) -> jnp.ndarray:
    """

    """
    return jnp.dot(kraus_opr, jnp.dot(rho, dagger(kraus_opr)))
