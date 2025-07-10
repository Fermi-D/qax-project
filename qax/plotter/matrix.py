import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d(density_mat: jax.Array, fig_size: tuple) -> None:
    '''

    '''
    # matplotlib settings
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Real part")
    ax2.set_title("Imaginary part")

    # aspect settings
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    # extract real&imaginary part in density matrix
    real_part = jnp.real(density_mat)
    imag_part = jnp.imag(density_mat)

    # plot real&imaginary part
    hm1 = ax1.pcolormesh(real_part, cmap="viridis")
    hm2 = ax2.pcolormesh(imag_part, cmap="cividis")

    # colorbar settings
    fig.colorbar(hm1, ax=ax1)
    fig.colorbar(hm2, ax=ax2)

    plt.tight_layout(rect=[0, 0.03, 0.75, 0.95])
    #plt.tight_layout()
    plt.show()

def plot_3d(density_mat: jax.Array, fig_size: tuple) -> None:
    '''

    '''
    # matplotlib settings
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_title("Real part")
    ax2.set_title("Imaginary part")

    # extract matrix size
    row_size = density_mat.shape[0]
    col_size = density_mat.shape[1]
    
    # extract real&imaginary part in density matrix
    real_part = jnp.real(density_mat)
    imag_part = jnp.imag(density_mat)

    # plot real&imaginary part
    for i in range(row_size):
        for j in range(col_size):
            ax1.bar3d(i, j, 0, 0.8, 0.8, real_part[i, j], color="tab:blue" if real_part[i, j] >= 0 else "tab:orange")
            ax2.bar3d(i, j, 0, 0.8, 0.8, imag_part[i, j], color="tab:purple" if imag_part[i, j] >= 0 else "tab:green")

    value_min = min(np.min(real_part), np.min(imag_part))
    value_max = max(np.max(real_part), np.max(imag_part))
    ax1.set_zlim(value_min, value_max)
    ax2.set_zlim(value_min, value_max)

    plt.tight_layout(rect=[0, 0.03, 0.75, 0.95])
    #plt.tight_layout()
    plt.show()