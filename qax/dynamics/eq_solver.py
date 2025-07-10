def Schrodinger(H: jnp.ndarray, vec_0: jnp.ndarray, start_time: float, end_time: float, time_step: float) -> jnp.ndarray:
    """
    Solve the Schrodinger equation.
    Args:
        H: system Hamiltonian
        jump_operators: list of jnp.ndarray, jump operators {L_k}.
        rho0: jnp.ndarray, initial density matrix.
        t_span: tuple, (t_start, t_end), the time range.
        dt: float, time step.
        hbar: float, Planck's constant divided by 2π (default: 1.0).
    
    Returns:
        t_vals: jnp.ndarray, array of time points.
        rho_t: jnp.ndarray, array of density matrices at each time point.
    """
    rhs = -1.0j * H
    equation = lambda t, vec, args: rhs
    ode = ODETerm(equation)
    integrator = Dopri5()
    saveat = SaveAt(ts=jnp.arange(start_time, end_time, time_step))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    solver = diffeqsolve(ode, integrator, t0=start_time, t1=end_time, dt0=step_time, y0=vec_0, saveat=saveat, stepsize_controller=stepsize_controller)
    
    return solver

def Lindblad(H: jnp.ndarray, rho_0: jnp.ndarray, jump_oprs: jnp.ndarray, start_time: float, end_time: float, time_step: float) -> jnp.ndarray:
    """
    Solve the Lindblad master equation.
    Args:
        H: system Hamiltonian
        jump_operators: list of jnp.ndarray, jump operators {L_k}.
        rho0: jnp.ndarray, initial density matrix.
        t_span: tuple, (t_start, t_end), the time range.
        dt: float, time step.
        hbar: float, Planck's constant divided by 2π (default: 1.0).
    
    Returns:
        t_vals: jnp.ndarray, array of time points.
        rho_t: jnp.ndarray, array of density matrices at each time point.
    """
    # 
    commutator = -1.0j * mt.commutator(H, rho_0)
    # 
    dissipator = jnp.zeros(H.shape, dtype=jnp.complex64)
    def sum_dissipator(i, rho) -> jnp.ndarray:
        return mt.kraus_repr(rho, jump_oprs) - 0.5*mt.anti_commutator(jnp.dot(mt.dagger(jump_oprs), jump_oprs), rho)
    dissipator = jax.lax.fori_loop(0, n_photon+1, sum_dissipator, rho_0)

    rhs = commutator + dissipator
    equation = lambda t, rho, args: rhs
    ode = ODETerm(equation)
    integrator = Dopri5()
    saveat = SaveAt(ts=jnp.arange(start_time, end_time, time_step))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    solver = diffeqsolve(ode, integrator, t0=start_time, t1=end_time, dt0=time_step, y0=rho_0, saveat=saveat, stepsize_controller=stepsize_controller)

    return solver

@jit
def Schrodinger(hamiltonian, psi0, time_duration, dt, hbar=1.0):
    """
    Solves the time-dependent Schrödinger equation (TDSE).
    
    Args:
        hamiltonian_fn: Callable, H(t), a function that returns the Hamiltonian at time t.
        psi0: jnp.array, initial state vector.
        t_span: tuple, (t_start, t_end) the time range.
        dt: float, time step.
        hbar: float, Planck's constant divided by 2π (default: 1.0).
        
    Returns:
        t_vals: jnp.array, array of time points.
        psi_t: jnp.array, array of state vectors at each time point.
    """
    def time_derivative(psi, t):
        H_t = hamiltonian(t)  # Evaluate the Hamiltonian at time t
        return -1j / hbar * jnp.dot(H_t, psi)
    
    # Time points
    t_start, t_end = t_span
    t_vals = jnp.arange(t_start, t_end, dt)
    
    # Initialize state vector storage
    psi_t = [psi0]
    
    # Time evolution using Runge-Kutta 4th order method
    for t in t_vals[:-1]:
        k1 = time_derivative(psi_t[-1], t)
        k2 = time_derivative(psi_t[-1] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = time_derivative(psi_t[-1] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = time_derivative(psi_t[-1] + dt * k3, t + dt)
        psi_next = psi_t[-1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        psi_t.append(psi_next)
    
    return t_vals, jnp.array(psi_t)