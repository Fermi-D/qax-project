import jax.numpy as jnp
from jax import jit
from functools import partial

def identity(dim: int) -> jnp.ndarray:
    """
    Returns the Pauli X (sigma_x) operator.
    The Pauli X operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]
    
    Returns:
        jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    """
    return jnp.identity(dim, dtype=jnp.complex64)

def pauli_x() -> jnp.ndarray:
    return jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)
    """
    Returns the Pauli X (sigma_x) operator.
    The Pauli X operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]
    
    Returns:
        jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    """
    #return jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)

def pauli_y() -> jnp.ndarray:
    """
    Returns the Pauli Y (sigma_x) operator.
    The Pauli Y operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]
    
    Returns:
        jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    """
    return jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)

def pauli_pls() -> jnp.ndarray:
    return jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64) # pauli_puls

def pauli_mns() -> jnp.ndarray: 
    return jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex64) # \sigma_-

def hadamard() -> jnp.ndarray:
    return 1/jnp.sqrt(2) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64)

def phase_s() -> jnp.ndarray:
    return

def phase_sdag() -> jnp.ndarray:
    return

def phase_t() -> jnp.ndarray:
    return

def phase_tdag() -> jnp.ndarray:
    return

def annihilate(dim: int) -> jnp.ndarray:
    # Args: dim (int): Dimension of Hilbert space
    # Returns: Tensor([N, N], complex): NxN annihilation operator
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=1)

def create(dim: int) -> jnp.ndarray:
    # Args: dim (int): Dimension of Hilbert space
    # Returns: Tensor([N, N], complex): NxN creation operator
    a = annihilate(dim)
    return jnp.conjugate(jnp.transpose(a))
    return jax.lax.transpose(jax.lax.conj(a), (1, 0))

def num(dim: int) -> jnp.ndarray:
    # Args: dim (int): Dimension of Hilbert space
    # Returns: Tensor([N, N], complex): NxN number operator
    return jnp.diag(jnp.arange(0, dim, dtype=jnp.complex64))

def position(dim: int) -> jnp.ndarray:
    # Args: dim (int): Dimension of Hilbert space
    # Returns: Tensor([N, N], complex): NxN position operator
    a_dag = create(dim)
    a = annihilate(dim)
    return (a+a_dag) / jnp.sqrt(2)

def momentum(dim: int) -> jnp.ndarray:
    # Args: dim (int): Dimension of Hilbert space
    # Returns: Tensor([N, N], complex): NxN momentum operator
    a_dag = create(dim)
    a = annihilate(dim)
    return (a-a_dag) / jnp.sqrt(2)

def displacement(alpha: jnp.complex64, dim: int) -> jnp.ndarray:
    """
    
    """
    return expm(alpha*create(dim) - jnp.conjugate(alpha)*annihilate(dim))

def squeeze(z: jnp.complex64, dim: int) -> jnp.ndarray:
    """
    
    """
    return expm((jnp.conjugate(z)/2)*jnp.dot(annihilate(dim), annihilate(dim)) - (z/2)*jnp.dot(create(dim), create(dim)))