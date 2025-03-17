import jax
import jax.numpy as jnp
from functools import partial
from . import gate

import importlib
importlib.reload(gate)

class QuantumCircuit:
    def __init__(self, n_qubits: int, state_class: str):
        self.init_state = jnp.zeros([1, 2**n_qubits], dtype=jnp.complex64).at[0, 0].set(1+0j)
        self.state_class = state_class
        
        if state_class == "state vector":
            self.init_state = jax.lax.transpose(self.init_state, (1, 0)) 
        
        elif stateclass == "density matrix":
            self.init_state = jnp.outer(self.init_state, self.init_state)
            
        else:
            raise ValueError("Invalid quantum state class provided. Expected 'state vector' or 'density matrix'.")
            
    @partial(jax.jit, static_argnums=(0, 1))
    def act(self, gate: dict, state=None) -> jnp.ndarray:
        self.gate_mat = gate["matrix"]
        self.label = gate.get("label", None)
        self.state = self.init_state if state is None else state
        
        if self.state_class == "state vector":
            print(self.gate_mat)
            self.state = jnp.dot(self.gate_mat, self.state)
            
        return self.state