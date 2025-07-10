import jax.numpy as jnp
from jax import jit
from functools import partial

def constant(amp, theta, duration):
    element = amp * jnp.exp(1.0j*theta)
    return jnp.full(element, duration)

def gaussian(amp
