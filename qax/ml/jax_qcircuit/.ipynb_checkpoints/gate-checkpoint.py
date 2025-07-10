import jax.numpy as jnp
from jax import jit
from functools import partial, reduce

import sys
sys.path.append("/home/users/u0001529/ondemand/NN-QST/jax_qcircuit")

import operators
importlib.reload(operators)

@partial(jit, static_argnums=(0, 1))
def X(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_x() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    return mat

@partial(jit, static_argnums=(0, 1))
def Y(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_y() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    return mat

@partial(jit, static_argnums=(0, 1))
def Z(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_z() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    return mat