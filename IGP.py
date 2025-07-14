import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def IGP(KIMdl, zetaini, A, XI, iter=30, eps=0.05):
    """
    PGIGP algorithm: tracks reference using GPR model and reachable sets A.

    Args:
        KIMdl: Trained GPytorch model for control prediction.
        zetaini: Initial condition array-like [z1, z2, u_prev].
        A: Reachable sets data structure.
        XI: Training data array-like.
        iter: Number of iterations.
        eps: Small constant for fallback control.

    Returns:
        yTest: np.ndarray of predicted outputs.
    """
    yTest = []
    zetaTest = [np.array(zetaini, dtype=np.float32)]
    zetacur = np.array(zetaini, dtype=np.float32)

    for t in range(iter):
        yref = search(zetacur, A, XI)  # Your previously defined search function

        if yref is None:
            # No reference found: fallback control
            u = (eps * (2 * np.random.rand() - 1)
                 - (1 + eps * (2 * np.random.rand() - 1)) * zetacur[0]
                 + (1 + eps * (2 * np.random.rand() - 1)) * zetacur[1]
                 - (1 + eps * (2 * np.random.rand() - 1)) * zetacur[2])
        else:
            print("IGP selected idx")
            # Use GP model to predict control
            input_tensor = torch.tensor([[yref, *zetacur]], dtype=torch.float32)
            with torch.no_grad():
                u = KIMdl(input_tensor).mean.item()

        # System dynamics: your sys function from sys_model
        y = sys(zetacur, u)
        zetacur = np.array([y, zetacur[0], u], dtype=np.float32)

        yTest.append(y)
        zetaTest.append(zetacur.copy())

    yTest = np.array(yTest)
    zetaTest = np.array(zetaTest)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=0, vmax=len(zetaTest))
    colors = cm.viridis(norm(range(len(zetaTest))))

    # Scatter points colored by time step
    ax.scatter(zetaTest[:, 0], zetaTest[:, 1], zetaTest[:, 2],
               c=range(len(zetaTest)), cmap='viridis', s=50)

    # Draw arrows between points with color gradient
    for i in range(len(zetaTest) - 1):
        start = zetaTest[i]
        vec = zetaTest[i + 1] - start
        ax.quiver(start[0], start[1], start[2],
                  vec[0], vec[1], vec[2],
                  color=colors[i], linewidth=1.5, arrow_length_ratio=0.15)

    # Annotate start and end points
    ax.text(*zetaTest[0], 'Start', color='green', fontsize=12)
    ax.text(*zetaTest[-1], 'End', color='red', fontsize=12)

    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('u_prev')
    ax.set_title('IGP Trajectory')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return yTest