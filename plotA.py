import matplotlib.pyplot as plt

def plotA(A_all, XI, delta_list, kbar,states_visited):
    """
    Plot reachable sets overlapped for specific projections,
    and overlay current states trajectory as blue dashed lines with markers.

    Parameters:
        A_all (dict): Dictionary mapping delta -> reachable sets A (list of dicts).
        XI (np.ndarray): Data array shape (n_samples, 4).
        delta_list (array-like): List or array of delta values.
        kbar (int): Maximum iteration depth.
        states_visited (np.ndarray): Array shape (N, 3) of tracked states (y(t), y(t-1), u(t-1)).
    """

    projections = [
        (1, 2),  # y(t) vs y(t-1)
        (2, 3),  # y(t-1) vs u(t-1)
        (3, 1)   # u(t-1) vs y(t)
    ]

    axis_labels = {
        0: r'$y(t+1)$',
        1: r'$y(t)$',
        2: r'$y(t-1)$',
        3: r'$u(t-1)$'
    }

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(delta_list))]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (dim_x, dim_y) in zip(axs, projections):
        ax.set_title(f'Projection: {axis_labels[dim_x]} vs {axis_labels[dim_y]}')
        ax.set_xlabel(axis_labels[dim_x])
        ax.set_ylabel(axis_labels[dim_y])
        ax.grid(True)

        # Plot reachable set balls for all deltas
        for color, delta in zip(colors, delta_list):
            A = A_all[delta]
            for k in range(min(len(A), kbar + 1)):
                indices = A[k]['i']
                if len(indices) == 0:
                    continue
                points = XI[indices][:, [dim_x, dim_y]]
                ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.3, color=color)

        # Plot current states trajectory in this projection
        # states_visited columns correspond to dims (1, 2, 3) matching y(t), y(t-1), u(t-1)
        # So index them accordingly for projection dims:
        states_proj_x = states_visited[:, dim_x - 1]  # dim_x-1 because states_visited has dims (y(t), y(t-1), u(t-1)) = (1,2,3)
        states_proj_y = states_visited[:, dim_y - 1]

        ax.plot(states_proj_x, states_proj_y, linestyle='--', color='blue', label='Current state')

        # Legend entries per delta (only add once)
        for color, delta in zip(colors, delta_list):
            ax.scatter([], [], color=color, label=f'delta={delta}')

        ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()