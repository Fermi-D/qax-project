import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Sequence
from functools import partial

from ...core import StateVector, Operator
from ...utils import device

class SchrodingerEquation:
    """
    Solves the time-dependent Schrödinger equation for a given quantum system.
    
    This class encapsulates the Hamiltonian, initial state, and time span,
    and provides methods to run the simulation and compute expectation values.
    """
    def __init__(
        self,
        hamiltonian: Operator | Callable[[float], Operator],
        initial_state: StateVector,
        t_span: tuple[float, float],
        hbar: float = 1.0,
        target_device: str | jax.Device | None = None
    ):
        """
        Initializes the solver for the Schrödinger equation.

        Args:
            hamiltonian (Operator | Callable): The system's Hamiltonian. 
                Can be a static Operator or a time-dependent function H(t).
            initial_state (StateVector): The initial state vector at t0.
            t_span (tuple[float, float]): The start and end time (t0, t1).
            hbar (float, optional): Planck's constant. Defaults to 1.0.
            target_device (str | jax.Device | None, optional): 
                The device to run the computation on ('cpu', 'gpu', etc.).
                If None, JAX's default device is used. Defaults to None.
        """
        # デバイスを取得し、インスタンス変数として保存
        self.device = device.select_device(target_device)
        
        # ハミルトニアンと初期状態を指定されたデバイスに移動
        if callable(hamiltonian):
            self.hamiltonian = hamiltonian  # A function is not moved
        else:
            self.hamiltonian = device.to(hamiltonian, self.device)
            
        self.initial_state = device.to(initial_state, self.device)
        self.t_span = t_span
        self.hbar = hbar
        
        # 計算結果を保持するための変数を初期化
        self.solution: diffrax.Solution | None = None

    def run(
        self,
        dt0: float = 0.01,
        saveat: diffrax.SaveAt | None = None,
        solver: diffrax.AbstractSolver = None,
        stepsize_controller: diffrax.AbstractStepSizeController = None
    ) -> diffrax.Solution:
        """
        Runs the time evolution solver.
        ... (docstringは変更なし) ...
        """
        t0, t1 = self.t_span
        
        if solver is None:
            solver = diffrax.Dopri5()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

        def vector_field(t, y, args):
            H_or_func, hbar_val = args
            # H_or_funcが静的なので、このif文もJITコンパイル可能になる
            if callable(H_or_func):
                H = H_or_func(t)
            else:
                H = H_or_func
            return (-1j / hbar_val) * (H.data @ y)

        term = diffrax.ODETerm(vector_field)

        # 指定されたデバイス上で計算を実行
        with jax.default_device(self.device):
            
            # ★ 修正点: JITデコレータに静的な引数を指定
            # functools.partialを使って、jax.jitに引数を渡す
            @partial(jax.jit, static_argnames=['H'])
            def solve(y0_data, H):
                return diffrax.diffeqsolve(
                    term,
                    solver,
                    t0,
                    t1,
                    dt0,
                    y0_data,
                    args=(H, self.hbar), # Hをargs経由で渡す
                    saveat=saveat,
                    stepsize_controller=stepsize_controller
                )
            
            # 計算を実行
            sol = solve(self.initial_state.data, self.hamiltonian)
            
        self.solution = sol
        return sol

    def expectation_value(self, observable: Operator) -> tuple[jax.Array, jax.Array]:
        """
        Computes the expectation value of an observable over time.
        The `run()` method must be called before this.
        """
        if self.solution is None:
            raise RuntimeError("Must call .run() before computing expectation values.")
        
        # observableを計算デバイスに移動
        obs_on_device = device.to(observable, self.device)

        # 期待値 <ψ|O|ψ> を計算する純粋関数
        @jax.vmap
        def _calculate_ev(psi_data):
            ket = StateVector(psi_data)
            bra_op = ket.dagger()
            exp_val = (bra_op @ obs_on_device @ ket).data[0, 0]
            return exp_val.real

        # JITコンパイルして高速化
        exp_vals = jax.jit(_calculate_ev)(self.solution.ys)
        
        return self.solution.ts, exp_vals

    def plot_expectation(self, observable: Operator, title: str | None = None, label: str | None = None):
        """
        Computes and plots the expectation value of an observable.
        """
        ts, evs = self.expectation_value(observable)
        
        plt.plot(np.asarray(ts), np.asarray(evs), label=label)
        plt.xlabel("Time")
        plt.ylabel("Expectation Value")
        if title:
            plt.title(title)
        plt.grid(True)
        if label:
            plt.legend()