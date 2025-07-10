from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import sympy
from IPython.display import display, Math
from jax.scipy.linalg import expm

def setup_printing():
    '''
    Initializes SymPy's pretty printing for Jupyter.
    '''
    sympy.init_printing()

def _display_resizable_matrix(sympy_matrix_obj):
    '''
    Helper function to display a SymPy matrix object in Jupyter
    with resizable brackets.
    
    Args:
        sympy_matrix_obj (sympy.Matrix): The SymPy matrix to display.
    '''
    latex_str = sympy.latex(sympy_matrix_obj)
    if r'\begin{bmatrix}' in latex_str:
        resizable_str = latex_str.replace(r'\begin{bmatrix}', r'\left[ \begin{matrix}') \
                                 .replace(r'\end{bmatrix}', r'\end{matrix} \right]')
    elif r'\begin{pmatrix}' in latex_str:
        resizable_str = latex_str.replace(r'\begin{pmatrix}', r'\left( \begin{matrix}') \
                                 .replace(r'\end{pmatrix}', r'\end{matrix} \right)')
    else:
        resizable_str = latex_str
    display(Math(resizable_str))

class StateVector:
    '''
    A wrapper class for a JAX array representing a quantum state vector.
    '''
    def __init__(self, array: jax.Array | np.ndarray):
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array, dtype=jnp.complex64)
        if array.ndim != 1:
            raise ValueError(f"StateVector must be initialized with a 1D array, but got shape {array.shape}")
        self.data = array

    def info(self) -> None:
        '''
        Displays key information about the state vector.
        '''
        print(f"Shape: {self.shape}")
        print(f"Dtype: {self.dtype}")
        print(f"Norm: {jnp.linalg.norm(self.data):.6f}")

        display_array = jnp.round(self.data, decimals=3)
        sympy_matrix = sympy.Matrix(np.asarray(display_array))
        _display_resizable_matrix(sympy_matrix)

    def _repr_latex_(self) -> str:
        '''
        Rich display in Jupyter, rounded to 3 decimal places.
        '''
        sympy_matrix = sympy.Matrix(np.asarray(jnp.round(self.data, 3)))
        latex_str = sympy.latex(sympy_matrix)
        resizable_str = latex_str.replace(r'\begin{bmatrix}', r'\left[ \begin{matrix}') \
                                 .replace(r'\end{bmatrix}', r'\end{matrix} \right]')
        return f"$${resizable_str}$$"

    def __repr__(self) -> str:
        return f"StateVector(dtype={self.dtype}, shape={self.shape})\n{self.data}"

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other: StateVector) -> StateVector:
        if isinstance(other, StateVector):
            return StateVector(self.data + other.data)
        return NotImplemented

    def __sub__(self, other: StateVector) -> StateVector:
        if isinstance(other, StateVector):
            return StateVector(self.data - other.data)
        return NotImplemented

    def __mul__(self, scalar) -> StateVector:
        return StateVector(self.data * scalar)
    
    def __rmul__(self, scalar) -> StateVector:
        return StateVector(scalar * self.data)

    def __truediv__(self, scalar) -> StateVector:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an Operator by zero.")
        return StateVector(self.data / scalar)

    def __rmatmul__(self, other_matrix: Operator):
        if isinstance(other_matrix, Operator):
            return StateVector(other_matrix.data @ self.data)
        return NotImplemented
    
    def dagger(self) -> Operator:
        '''
        Returns the Hermitian conjugate (bra) as a 1xN Operator.
        '''
        return Operator(jnp.conj(self.data).reshape(1, -1))

class Operator:
    '''
    A wrapper class for a JAX array representing a quantum operator (matrix).
    '''
    def __init__(self, array: jax.Array | np.ndarray):
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array, dtype=jnp.complex64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"Operator must be initialized with a square 2D array, but got shape {array.shape}")
        self.data = array

    def info(self) -> None:
        '''
        Displays key information about the operator.
        Displayed values are rounded to 3 decimal places.
        '''
        print(f"Shape: {self.shape}")
        print(f"Dtype: {self.dtype}")
        print(f"Hermitian: {self.is_hermitian()}")

        display_array = jnp.round(self.data, decimals=3)
        sympy_matrix = sympy.Matrix(np.asarray(display_array))
        _display_resizable_matrix(sympy_matrix)

    def _repr_latex_(self) -> str:
        '''
        Rich display in Jupyter, rounded to 3 decimal places.
        '''
        sympy_matrix = sympy.Matrix(np.asarray(jnp.round(self.data, 3)))
        latex_str = sympy.latex(sympy_matrix)
        resizable_str = latex_str.replace(r'\begin{bmatrix}', r'\left[ \begin{matrix}') \
                                 .replace(r'\end{bmatrix}', r'\end{matrix} \right]')
        return f"$${resizable_str}$$"

    def __repr__(self) -> str:
        return f"Operator(dtype={self.dtype}, shape={self.shape})\n{self.data}"

    def dagger(self) -> Operator:
        return Operator(jnp.conj(jnp.transpose(self.data)))

    def trace(self) -> jax.Array:
        return jnp.trace(self.data)

    def is_hermitian(self, tol: float = 1e-6) -> bool:
        return jnp.allclose(self.data, self.dagger().data, atol=tol)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return Operator(self.data + other.data)
        return NotImplemented

    def __sub__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return Operator(self.data - other.data)
        return NotImplemented
        
    def __mul__(self, scalar) -> Operator:
        return Operator(self.data * scalar)

    def __rmul__(self, scalar) -> Operator:
        return Operator(scalar * self.data)

    def __truediv__(self, scalar) -> Operator:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an Operator by zero.")
        return Operator(self.data / scalar)

    def __matmul__(self, other: Operator | StateVector) -> Operator | StateVector:
        if isinstance(other, Operator):
            return Operator(self.data @ other.data)
            
        elif isinstance(other, StateVector):
            return StateVector(self.data @ other.data)
            
        return NotImplemented