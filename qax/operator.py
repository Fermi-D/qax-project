import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from .core import Operator

def identity(dim: int) -> jax.Array:
    '''
    Returns the identity operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns: 
      jnp.array: A 'dim'x'dim' matrix of the identity operator with complex64 precision.
    '''
    mat = jnp.identity(dim, dtype=jnp.complex64)
    return Operator(mat)

def pauli_x() -> jax.Array:
    '''
    Returns the Pauli X operator.
    The Pauli X operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    '''
    mat = jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)
    return Operator(mat)
    
def pauli_y() -> jax.Array:
    '''
    Returns the Pauli Y operator.
    The Pauli Y operator is a 2x2 matrix defined as:
    [[0, -i],
     [i, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli Y operator with complex64 precision.
    '''
    mat = jnp.array([[0+0j, 0-1j], [0+1j, 0+0j]], dtype=jnp.complex64)
    return Operator(mat)

def pauli_z() -> jax.Array:
    '''
    Returns the Pauli Z operator.
    The Pauli Z operator is a 2x2 matrix defined as:
    [[1, 0],
     [0, -1]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli Z operator with complex64 precision.
    '''
    mat = jnp.array([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=jnp.complex64)
    return Operator(mat)

def raising() -> jax.Array:
    '''
    Returns the raising operator.
    The raising operator is a 2x2 matrix defined as:
    [[0, 1],
     [0, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the raising operator with complex64 precision.
    '''
    mat = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64)
    return Operator(mat)

def lowering() -> jax.Array: 
    '''
    Returns the lowering operator.
    The lowering operator is a 2x2 matrix defined as:
    [[0, 0],
     [1, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the lowering operator with complex64 precision.
    '''
    mat = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex64)
    return Operator(mat)

def annihilation(dim: int) -> jax.Array:
    '''
    Returns the annihilation operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns:
      jnp.array: A 'dim'x'dim' matrix of the annihilation operator with complex64 precision.
    '''
    mat = jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=1)
    return Operator(mat)

def creation(dim: int) -> jax.Array:
    '''
    Returns the creation operator

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns:
      jnp.array: A 'dim'x'dim' matrix of the creation operator with complex64 precision.
    '''
    mat = jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=-1)
    return Operator(mat)

def number(dim: int) -> jax.Array:
    '''
    Returns the number operator.
    
    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      jnp.array: A 'dim'x'dim' matrix of the number operator with complex64 precision.
    '''
    mat = jnp.diag(jnp.arange(0, dim, dtype=jnp.complex64))
    return Operator(mat)

def position(dim: int) -> jnp.ndarray:
    '''
    Returns the position quadrature operator (dimensionless).
    Defined as x = (a + a_dag) / sqrt(2).

    Args:
        dim (int): The dimension of Hilbert space.

    Returns:
        jax.Array: A 'dim'x'dim' matrix of the position operator.
    '''
    a_dag = creation(dim)
    a = annihilation(dim)
    return (a+a_dag) / jnp.sqrt(2)

def momentum(dim: int) -> jnp.ndarray:
    '''
    Returns the momentum quadrature operator (dimensionless).
    Defined as p = i * (a_dag - a) / sqrt(2) to be Hermitian.

    Args:
        dim (int): The dimension of Hilbert space.

    Returns:
        jax.Array: A 'dim'x'dim' matrix of the momentum operator.
    '''
    a_dag = creation(dim)
    a = annihilation(dim)
    return 1j*(a_dag - a) / jnp.sqrt(2)

def displacement(alpha: complex, dim: int) -> jnp.ndarray:
    '''
    Returns the displacement operator D(alpha).
    This operator displaces a quantum state in phase space,
    generating a coherent state when applied to the vacuum.
    D(alpha) = exp(alpha * a_dag - alpha_conj * a)

    Args:
        alpha (complex): The complex displacement amplitude.
        dim (int): The dimension of Hilbert space.

    Returns:
        jax.Array: The 'dim'x'dim' displacement operator.
    '''
    a_dag = creation(dim)
    a = annihilation(dim)
    return expm(alpha*a_dag - jnp.conjugate(alpha)*a)

def squeeze(z: complex, dim: int) -> jnp.ndarray:
    '''
    Returns the squeeze operator S(z).
    This operator squeezes a quantum state in phase space,
    generating a squeezed state when applied to the vacuum.
    S(z) = exp(0.5 * (z_conj * a^2 - z * (a_dag)^2))

    Args:
        z (complex): The complex squeeze factor.
        dim (int): The dimension of Hilbert space.

    Returns:
        jax.Array: The 'dim'x'dim' squeeze operator.
    '''
    a_dag = creation(dim)
    a = annihilation(dim)
    return expm((jnp.conjugate(z)/2)*jnp.dot(a, a) - (z/2)*jnp.dot(a_dag, a_dag))