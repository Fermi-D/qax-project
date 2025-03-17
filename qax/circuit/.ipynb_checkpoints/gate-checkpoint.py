import jax.numpy as jnp
from jax import jit
from functools import partial, reduce

from . import operator

import importlib
importlib.reload(operator)

# 1-qubit gate
@partial(jit, static_argnums=(0, 1))
def X(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operator.pauli_x() if i == target_qubit_idx else operator.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"X_{target_qubit_idx}"
    return {"label": label, "matrix": mat}

@partial(jit, static_argnums=(0, 1))
def Y(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operator.pauli_y() if i == target_qubit_idx else operator.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"Y_{target_qubit_idx}"
    return mat

@partial(jit, static_argnums=(0, 1))
def Z(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operator.pauli_z() if i == target_qubit_idx else operator.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"Z_{target_qubit_idx}"
    return mat

@partial(jit, static_argnums=(0, 1))
def H(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operator.hadamard() if i == target_qubit_idx else operator.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"H_{target_qubit_idx}"
    return mat

# 2-qubit gate
@partial(jit, static_argnums=(0, 1))
def CNOT(controll_qubit_idx: int, target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operator.pauli_x() if i == target_qubit_idx else operator.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"CNOT_{controll_qubit_idx}{target_qubit_idx}"
    return mat, label