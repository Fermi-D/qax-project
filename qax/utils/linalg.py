import jax
import jax.numpy as jnp
from ..core import StateVector, Operator

def kron_prod(A: Operator | StateVector, B: Operator | StateVector) -> Operator | StateVector:
    '''
    Computes the Kronecker product of two quantum objects (Operators or StateVectors).
    - Operator otimes Operator -> new Operator
    - StateVector otimes StateVector -> new composite StateVector
    - Operator otimes StateVector -> new Operator
    - StateVector otimes Operator -> new Operator

    Args:
        A (Operator | StateVector): First input object.
        B (Operator | StateVector): Second input object.
    
    Returns:
        Operator | StateVector: The Kronecker product, returned as a new Operator or StateVector based on the input types.
    '''
    data1 = A.data
    data2 = B.data

    result = jnp.kron(data1, data2)

    if isinstance(A, StateVector) and isinstance(B, StateVector):
        return StateVector(result)
    else:
        return Operator(result)

def mat_prod(A: Operator, B: Operator) -> Operator:
    '''
    Computes the Kronecker product of two operators.
    - 
    '''

def normalize(state: StateVector | Operator) -> StateVector | Operator:
    """
    Normalizes a state vector (to norm=1) or a density matrix (to trace=1).

    Args:
        state (StateVector | Operator): Input state to be normalized.

    Returns:
        StateVector | Operator: Normalized state.
    """
    if isinstance(state, StateVector):
        # State vector normalization
        norm = jnp.linalg.norm(state.data)
        if norm == 0:
            raise ValueError("State vector has zero norm and cannot be normalized.")
       
        return state / norm
    
    elif isinstance(state, Operator):
        # Density matrix normalization
        trace = jnp.trace(state.data)
        if jnp.abs(trace) < 1e-9:
            raise ValueError("Density matrix has zero trace and cannot be normalized.")

        return state / trace
    
    else:
        raise TypeError(f"Input must be a StateVector or Operator, but got {type(state)}")

def commutator(A: Operator, B: Operator) -> Operator:
    """
    Computes the commutator of two operators, [A, B] = AB - BA.

    Args:
        op1 (Operator): The first operator (A).
        op2 (Operator): The second operator (B).

    Returns:
        Operator: The commutator [A, B].
    """
    return A@B - B@A

def anti_commutator(op1: Operator, op2: Operator) -> Operator:
    """
    Computes the anti-commutator of two operators, {A, B} = AB + BA.

    Args:
        op1 (Operator): The first operator (A).
        op2 (Operator): The second operator (B).

    Returns:
        Operator: The anti-commutator {A, B}.
    """
    return A@B + B@A

def vec2dm(state_vec: StateVector) -> Operator:
    """
    Converts a state vector |psi> into a density matrix rho = |psi><psi|.

    Args:
        state_vec (StateVector): A StateVector object representing |psi>.

    Returns:
        Operator: An Operator object representing the density matrix rho.
    """
    vec = state_vec.data
    mat = jnp.outer(vec, jnp.conjugate(vec))
    return Operator(mat)

def vec2bloch(state_vec: StateVector) -> jax.Array:
    """
    Calculates the Bloch vector [x, y, z] from a 2-level state vector.

    Args:
        state_vec (StateVector): A StateVector object of shape (2,).
    
    Returns:
        jax.Array: A 1D array of shape (3,) for the Bloch vector.
    """
    if state_vec.shape != (2,):
        raise ValueError("Input state_vec for Bloch vector must be of dimension 2.")

    # extract coefficients alpha, beta
    alpha = state_vec[0]
    beta = state_vec[1]

    # compute each elements of bloch vector 
    x = 2 * jnp.real(jnp.conjugate(alpha) * beta)
    y = 2 * jnp.imag(jnp.conjugate(alpha) * beta)
    z = jnp.abs(alpha)**2 - jnp.abs(beta)**2

    return jnp.array([x, y, z])

"""
def normalize(state) -> jax.Array:
    '''
    Normalize a state vector or a density matrix.
    
    Args:
        state (jnp.array): Input state, either a vector (1D) or a density matrix (2D).
    
    Returns:
        jnp.array: Normalized state vector or density matrix.
    
    Raises:
        ValueError: If the input dimensions are not 1 (vector) or 2 (matrix),
                    or if the norm or trace is zero.
    '''
    dim = jnp.ndim(state)

    # State vector normalization
    if dim == 1: 
        norm = jnp.linalg.norm(state)
        if norm == 0:
            raise ValueError(f"State vector has zero norm and cannot be normalized.")
        return state / norm

    # Density matrix normalization
    elif dim == 2: 
        trace = jnp.trace(state)
        if trace == 0:
            raise ValueError(f"Density matrix has zero trace and cannot be normalized.")
        return state / trace
    
    else:
        raise ValueError(f"Input dimensions do not match the expected dimensions. "
                         f"Got {dim}, but expected 1 (vector) or 2 (matrix).")

def commutator(A: jax.Array, B: jax.Array) -> jax.Array:
    '''
    Computes the commutator of two matrices, [A, B] = AB - BA.

    Args:
        A (jax.Array): The first matrix.
        B (jax.Array): The second matrix.

    Returns:
        jax.Array: The commutator [A, B].
    '''
    return jnp.dot(A, B) - jnp.dot(B, A)

def anti_commutator(A: jax.Array, B: jax.Array) -> jax.Array:
    '''
    Computes the anti-commutator of two matrices, {A, B} = AB + BA.

    Args:
        A (jax.Array): The first matrix.
        B (jax.Array): The second matrix.

    Returns:
        jax.Array: The anti-commutator {A, B}.
    '''
    return jnp.dot(A, B) + jnp.dot(B, A)

def vec2dm(state_vec: jax.Array) -> jax.Array:
    '''
    Converts a state vector |psi> into a density matrix rho = |psi><psi|.

    Args:
        state_vec (jax.Array): A 1D array representing the state vector.

    Returns:
        jax.Array: A 2D array representing the density matrix.
    '''
    if state_vec.ndim != 1:
        raise ValueError("Input state_vec must be a 1D array.")
        
    return jnp.outer(state_vec, jnp.conj(jnp.transpose(state_vec)))

def vec2bloch(state_vec: jax.Array) -> jax.Array:
    '''
    Calculates the Bloch vector [x, y, z] from a 2-level state vector.
    The components are the expectation values <sigma_x>, <sigma_y>, <sigma_z>.

    Args:
        state_vec (jax.Array): A 1D array of shape (2,) representing the state vector.
    
    Returns:
        jax.Array: A 1D array of shape (3,) for the Bloch vector [x, y, z].
    '''
    # extract coefficients alpha, beta
    alpha = state_vec[0]
    beta = state_vec[1]
    
    # compute expectation values
    x = 2 * jnp.real(jnp.conj(alpha) * beta)
    y = 2 * jnp.imag(jnp.conj(alpha) * beta)
    z = jnp.abs(alpha)**2 - jnp.abs(beta)**2

    return jnp.array([x, y, z])
"""