import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.numpy.linalg import matrix_power

from jax import jit
from functools import partial

import sys
sys.path.append("/home/users/u0001529/ondemand/SCQC_pulse-simulator/my_package")

import quantum_operator as qo
import math_tool

import importlib
importlib.reload(qo)
importlib.reload(math_tool)

def n_qubit(n: int) -> jnp.ndarray:
    '''
    Generate a n-qubit state |00...0> in a Hilbert space of dimension "2**n".
    Args:
        n (int): number of qubit
        
    Returns:
        jnp.ndarray: A 1D array representing the n-qubit state |00...0>.
    '''
    vec = jnp.zeros([1, 2**n], dtype=jnp.complex64)
    vec = vec.at[0, 0].set(1+0j)
    vec = jax.lax.transpose(vec, (1, 0))
    
    return vec
    
def vaccum(dim: int) -> jnp.ndarray:
    '''
    Generate a vaccum state |0> in a Hilbert space of dimension "dim".
    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the vaccum state |0>.
    '''
    vec = jnp.zeros([1, dim], dtype=jnp.complex64)
    vec = vec.at[0, 0].set(1+0j)
    vec = jax.lax.transpose(vec, (1, 0))
    
    return vec

def fock(n_photon: int, dim: int) -> jnp.ndarray:
    """
    Generate a fock state |n> in a Hilbert space of dimension `dim`.

    Args:
        n (int): The fock state index (0-based).
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the Fock state |n>.
    """
    def body(i, vec) -> jnp.ndarray:
        return jnp.dot(qo.create(dim), vec)
        
    return jax.lax.fori_loop(1, n_photon+1, body, vaccum(dim))
    
def coherent(alpha: complex, dim: int) -> jnp.ndarray:
    """
    Generate a coherent state |alpha> in a Hilbert space of dimension `dim`.

    Args:
        alpha (complex): 
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the coherent state |alpha>.
    """
    return jnp.dot(qo.displacement(alpha, dim), vaccum(dim))

def squeezed(z: float, dim: int) -> jnp.ndarray:
    """
    Generate a squeezed state |alpha> in a Hilbert space of dimension `dim`.

    Args:
        z (complex): 
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array representing the squeezed state |alpha>.
    """
    return jnp.dot(qo.squeeze(z, dim), vaccum(dim))

def position(x: float, dim: int) -> jnp.ndarray:
    """
    Generate a position state |x> in a Hilbert space of dimension `dim`.

    Args:
        x (complex): 
        dim (int): The dimension of the Hilbert space.
    """
    coff_1 = jnp.pi**(1/4)
    coff_2 = jnp.exp(0.5 * x**2)
    ops = coff_1 * coff_2 * expm(-0.5*matrix_power(qo.create(dim)-jnp.sqrt(2)*x*qo.identity(dim), 2))
    return jnp.dot(ops, vaccum(dim))

def momentum(p: float, dim: int) -> jnp.ndarray:
    """
    Generate a position state |p> in a Hilbert space of dimension `dim`.

    Args:
        x (complex):
        dim (int): The dimension of the Hilbert space.
        
    """
    return 0