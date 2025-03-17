import jax.numpy as jnp
from jax import jit
from functools import partial, reduce

from .. import operators

import importlib
importlib.reload(operator)

# 1-qubit gate
class X:
    def __init__(self, target_qubit_idx: int, n_qubits: int):
        self.target_qubit_idx = target_qubit_idx
        self.n_qubits = n_qubits
    def get_mat():
        

@partial(jit, static_argnums=(0, 1))
def X(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_x() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"X_{target_qubit_idx}"
    return {"label": label, "matrix": mat}

@partial(jit, static_argnums=(0, 1))
def Y(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_y() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"Y_{target_qubit_idx}"
    return mat

@partial(jit, static_argnums=(0, 1))
def Z(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_z() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"Z_{target_qubit_idx}"
    return mat

@partial(jit, static_argnums=(0, 1))
def H(target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.hadamard() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"H_{target_qubit_idx}"
    return mat

# 2-qubit gate
@partial(jit, static_argnums=(0, 1))
def CNOT(controll_qubit_idx: int, target_qubit_idx: int, n_qubits: int) -> jnp.ndarray:
    ops = [operators.pauli_x() if i == target_qubit_idx else operators.identity(dim=2) for i in range(n_qubits)]
    mat = reduce(jnp.kron, ops)
    label = f"CNOT_{controll_qubit_idx}{target_qubit_idx}"
    return mat, label