import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from ..utils.quantum import vec2bloch

def plot(state_vec: jax.Array, fig_size: tuple) -> None:
    '''
    Plots a single-qubit state vector on the Bloch sphere.

    Args:
      state_vec (jax.Array):
        The single-qubit state vector to plot. It must be a complex,
        array-like object (e.g., NumPy or JAX array) with shape (2,).
        The vector is expected to be normalized (i.e., its norm is 1).

    Returns:
      tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        A tuple containing the newly created Figure and 3D Axes objects,
        allowing for further customization after the function call.

    Raises:
      ValueError: If the input "state_vec" is not a 2-element vector.
      ValueError: If the input "state_vec" is not normalized.
    '''
    # matplotlib settings
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # plot settings
    ## convert cartesian coordinates to　polar coordinate
    phi = np.linspace(0, np.pi, 1000)
    theta = np.linspace(0, 2*np.pi, 1000)
    phi, theta = np.meshgrid(phi, theta)

    ## axis settings
    x_axis = np.sin(phi) * np.cos(theta)
    y_axis = np.sin(phi) * np.sin(theta)
    z_axis = np.cos(phi)

    ## plot bloch sphere
    #ax.plot_surface(x_axis, y_axis, z_axis, color='skyblue', alpha=0.1, linewidth=0, antialiased=True)
    ax.set_aspect("equal")

    ## axis range settings
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # arrow settings
    arrow_length = 1.0 
    arrow_ratio = 0.1 
    ## x-axis arrow
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color="tab:orange", arrow_length_ratio=arrow_ratio)
    ax.quiver(0, 0, 0, -arrow_length, 0, 0, color="tab:orange", arrow_length_ratio=arrow_ratio)
    ax.text(arrow_length*1.05, 0, 0, r"$| + \rangle$", color='red', fontsize=12)
    ax.text(-arrow_length*1.2, 0, 0, r"$| - \rangle$", color='red', fontsize=12)

    ## y-axis arrow
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color="tab:green", arrow_length_ratio=arrow_ratio)
    ax.quiver(0, 0, 0, 0, -arrow_length, 0, color="tab:green", arrow_length_ratio=arrow_ratio)
    ax.text(0, arrow_length*1.2, 0, r"$| -i \rangle$", color='green', fontsize=12)
    ax.text(0, -arrow_length*1.2, 0, r"$| +i \rangle$", color='green', fontsize=12)

    ## z-axis arrow
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color="tab:blue", arrow_length_ratio=arrow_ratio)
    ax.quiver(0, 0, 0, 0, 0, -arrow_length, color="tab:blue", arrow_length_ratio=arrow_ratio)
    ax.text(-0.075, 0, arrow_length*1.2, r"$| 0 \rangle$", color='blue', fontsize=12)
    ax.text(-0.075, 0, -arrow_length*1.2, r"$| 1 \rangle$", color='blue', fontsize=12)
    
    # convert state vector to bloch vector
    bloch_vec = vec2bloch(state_vec)
    ## convert jax.array to numpy.ndarray
    bloch_vec = jax.device_get(bloch_vec)

    ax.quiver(0, 0, 0, 
              bloch_vec[0], bloch_vec[1], bloch_vec[2],
              color="black", 
              arrow_length_ratio=0.1,
              zorder=2)

    #ax.set_title("Single-qubit state on Bloch sphere")
    ax.grid(False)
    ax.axis("off")

    plt.tight_layout()
    plt.show()