import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.numpy.linalg import matrix_power
from .core import StateVector, Operator
from . import operator as op

def n_qubit(n: int) -> jax.Array:
    '''
    Generate a n-qubit state |00...0> in a Hilbert space of dimension "2**n".
    Args:
        n (int): number of qubit
        
    Returns:
        jnp.ndarray: A 1D array representing the n-qubit state |00...0>.
    '''
    mat = jnp.zeros(2**n, dtype=jnp.complex64).at[0].set(1.0)
    return StateVector(mat)
    
def vacuum(dim: int) -> jax.Array:
    '''
    Generate a vaccum state |0> in a Hilbert space of dimension "dim".
    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the vaccum state |0>.
    '''
    mat = jnp.zeros(dim, dtype=jnp.complex64).at[0].set(1.0)
    return StateVector(mat)

def fock(n_photon: int, dim: int) -> jax.Array:
    '''
    Generate a fock state |n> in a Hilbert space of dimension `dim`.

    Args:
        n (int): The fock state index (0-based).
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the Fock state |n>.
    '''
    if n_photon >= dim:
        raise ValueError(f"n_photon ({n_photon}) must be less than the dimension ({dim}).")

    mat = jnp.zeros(dim, dtype=jnp.complex64).at[n_photon].set(1.0)
    return StateVector(mat)
    
def coherent(alpha: complex, dim: int) -> jax.Array:
    '''
    Generate a coherent state |alpha> in a Hilbert space of dimension `dim`.

    Args:
        alpha (complex): 
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the coherent state |alpha>.
    '''
    mat = jnp.dot(op.displacement(alpha, dim), vaccum(dim))
    return StateVector(mat)

def squeezed(z: complex, dim: int) -> jax.Array:
    '''
    Generate a squeezed state |alpha> in a Hilbert space of dimension `dim`.

    Args:
        z (complex): 
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the squeezed state |alpha>.
    '''
    mat = jnp.dot(op.squeeze(z, dim), vaccum(dim))
    return StateVector(mat)

def position(x: float, dim: int) -> jax.Array:
    '''
    Generate a position state |x> in a Hilbert space of dimension `dim`.

    Args:
        x (complex): 
        dim (int): The dimension of the Hilbert space.
    '''
    coff_1 = jnp.pi**(1/4)
    coff_2 = jnp.exp(0.5 * x**2)
    ops = coff_1 * coff_2 * expm(-0.5*matrix_power(op.creation(dim)-jnp.sqrt(2)*x*op.identity(dim), 2))
    mat = jnp.dot(ops, vaccum(dim))
    return StateVector(mat)

def momentum(p: float, dim: int) -> jax.Array:
    '''
    Generate a position state |p> in a Hilbert space of dimension `dim`.

    Args:
        p (complex):
        dim (int): The dimension of the Hilbert space.
    '''
    coff_1 = jnp.pi**(1/4)
    coff_2 = jnp.exp(0.5 * p**2)
    ops = coff_1 * coff_2 * expm(-0.5 * matrix_power(op.annihilation(dim) + 1j*jnp.sqrt(2)*p*op.identity(dim), 2))
    mat = jnp.dot(ops, vacuum(dim))
    return StateVector(mat)