import jax.numpy as jnp

import sys
sys.path.append("/home/users/u0001529/ondemand/NN-QST/jax_qcircuit")

import gpu_utils
import importlib
importlib.reload(qo)
importlib.reload(math_tool)

def kron_prod(A: jnp.ndarray, B: jnp.ndarray, gpu: bool) -> jnp.ndarray:
    gpu_true = pmap(lambda A, B: jnp.kron(A, B), )
    gpu_false = lambda A, B: jnp.kron(A, B)

    result = jax.lax.cond(gpu, gpu_true, gpu_false, *(A, B))
    return result