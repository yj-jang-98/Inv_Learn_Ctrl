import matplotlib.pyplot as plt
import numpy as np

def plotTrain(traj1True, traj1Meas, traj2True, traj2Meas, dt):
    # --- Plot Output Trajectories ---
    plt.figure(figsize=(10, 5))

    # --- Trajectory 1 ---
    t1 = np.arange(len(traj1True)) * dt
    plt.plot(t1, traj1True, 'g-', label='Trajectory 1 True (init = +2.0)')
    plt.plot(t1, traj1Meas, 'g--', label='Trajectory 1 Measured (noisy)')

    # --- Trajectory 2 ---
    t2 = np.arange(len(traj2True)) * dt
    plt.plot(t2, traj2True, 'r-', label='Trajectory 2 True (init = -2.0)')
    plt.plot(t2, traj2Meas, 'r--', label='Trajectory 2 Measured (noisy)')

    # --- Formatting ---
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Output y(t)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plotA(A_all, XI, Delta, kbar, states_sims):
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import matplotlib.ticker as mticker

    # --- Global LaTeX-style formatting ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 16,
    })

    # --- Define projections ---
    projections = [
        (1, 2),  # y(t) vs y(t-1)
        (2, 3),  # y(t-1) vs u(t-1)
        (3, 1)   # u(t-1) vs y(t)
    ]

    # --- Axis labels for each dimension ---
    axis_labels = {
        1: r'$y(t)$',
        2: r'$y(t-1)$',
        3: r'$u(t-1)$'
    }

    # --- Equilibrium point ---
    eq_point = np.array([0.0, 0.0, 1/3])

    # --- Plot figure ---
    fig, axs = plt.subplots(3, 1, figsize=(6, 7.5))

    # --- Styles ---
    trajectory_colors = ['#4682B4', '#556B2F', "#C63F3F"]    # blue, green, red
    trajectory_labels = [
        r'$\zeta(0)=[2;2;0]$',
        r'$\zeta(0)=[-1;-1;0]$',
        r'$\zeta(0)=[1;1;0]$'
    ]
    trajectory_markers = ['s', 'o', '^']
    trajectory_linestyles = ['-', '--', ':']

    # --- Plot loop ---
    for i, (ax, (dim_x, dim_y)) in enumerate(zip(axs, projections)):
        # --- Formatting of each subplot ---
        ax.set_xlabel(axis_labels[dim_x])
        ax.set_ylabel(axis_labels[dim_y])
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # --- Plot A_\delta^j ---
        for delta in Delta:
            A = A_all[delta]

            # --- Keep track of already drawn circles ---
            drawn_circles = []
            # --- Don't draw new circle if it overlaps with already drawn circles ---
            circles_to_draw = []  

            for k in range(min(len(A), kbar + 1)):
                indices = A[k]['i']
                radii = A[k]['ri']
                if len(indices) == 0:
                    continue

                # --- Project onto current subplot dimensions ---
                centers = XI[indices][:, [dim_x, dim_y]]

                for center, r in zip(centers, radii):
                    # --- Skip if fully contained in an already drawn circle ---
                    inside_any = any(
                        (np.linalg.norm(center - c_center) + r) <= c_r
                        for c_center, c_r in drawn_circles
                    )
                    if not inside_any:
                        circles_to_draw.append(patches.Circle(center, radius=r))
                        drawn_circles.append((center, r))

            # --- Add all circles to subplot ---
            if circles_to_draw:
                collection = PatchCollection(
                    circles_to_draw, facecolor="#B9B9B9", edgecolor="#B9B9B9",
                    alpha=1, linewidths=1
                )
                ax.add_collection(collection)

        # --- Plot simulated trajectories ---
        for states, t_color, t_label, t_marker, t_linestyle in zip(
            states_sims[3:6], trajectory_colors, trajectory_labels, trajectory_markers, trajectory_linestyles
        ):
            # --- Project trajectory onto subplot dimensions ---
            x_vals = states[:, dim_x - 1]
            y_vals = states[:, dim_y - 1]

            # --- Plot full trajectory ---
            ax.plot(x_vals, y_vals, linestyle=t_linestyle, color=t_color, linewidth=2.5)

            # --- Highlight initial point with marker ---
            ax.scatter(
                x_vals[0], y_vals[0], color=t_color, marker=t_marker, s=90,
                edgecolor='black', label=t_label, zorder=5
            )

        # --- Plot equilibrium point ---
        eq_x, eq_y = eq_point[dim_x - 1], eq_point[dim_y - 1]
        ax.scatter(
            eq_x, eq_y, color=(0, 1, 1), marker='o', edgecolor='black',
            s=60, label='Equilibrium', zorder=6
        )

        # --- Only show legend on the last subplot (to avoid repetition) ---
        if i == 2:
            ax.legend(
                loc='lower right', ncol=1, borderpad=0.3, handletextpad=0.2,
                prop={'family': 'serif', 'size': 12}
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("traj.pdf")
    plt.show()


def plotResult(traj1Meas, traj2Meas, trajectory_sims, simTimesteps,offlineStd,onlineStd):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    # --- Configure LaTeX-style fonts globally ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 24,
    })

    # --- Add zoomed inset ---
    def add_inset(ax, ys, t0=100, t1=150, colors=None, styles=None, markers=None):
        """Add zoom-in inset for given trajectories ys onto axis ax."""
        n = len(ys[0])
        i0, i1 = max(0, t0), min(t1, n - 1)
        x_zoom = np.arange(i0, i1 + 1)

        axins = inset_axes(
            ax, width="33%", height="40%",
            bbox_to_anchor=(-0.02, 0.02, 1, 1),
            bbox_transform=ax.transAxes,
            loc="lower right", borderpad=0
        )

        # --- Plot zoomed segments (no markers inside inset for clarity) ---
        for y, c, s in zip(ys, colors, styles):
            axins.plot(x_zoom, y[t0:t1+1], linestyle=s, linewidth=2.2, color=c)

        # Dynamic y-limits
        lo = min(np.min(y[t0:t1+1]) for y in ys)
        hi = max(np.max(y[t0:t1+1]) for y in ys)
        pad = 0.05 * max(1e-9, hi - lo)
        axins.set_xlim(t0, t1)
        axins.set_ylim(lo - pad, hi + pad)

        # Cosmetics
        axins.grid(True, linewidth=0.8)
        axins.tick_params(axis="y", labelsize=16)
        for spine in axins.spines.values():
            spine.set_linewidth(1.5)
        axins.set_xticks([])
        axins.set_xticklabels([])

        # Connector lines
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5", linewidth=1.5)

    # --- Styles ---
    colors = ["#4682B4", "#556B2F", "#C63F3F"]
    styles = ["-", "--", ":"]
    markers = ["s", "o", "^"]  # square, circle, triangle
    labels = [
        r"$\zeta(0)=[2;2;0]$",
        r"$\zeta(0)=[-1;-1;0]$",
        r"$\zeta(0)=[1;1;0]$"
    ]

    # --- Plot figure ---
    fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # --- Top subplot: training data ---
    axs[0].plot(traj1Meas, "--", color="black", linewidth=3,
                label=r"$(K_p,K_I)=(0,2),~\zeta(0)=[2;2;0]$")
    axs[0].plot(traj2Meas, ":", color="purple", linewidth=3,
                label=r"$(K_p,K_I)=(0.2,1),~\zeta(0)=[-1;-1;0]$")
    axs[0].set_ylabel(r"$y(t)$")
    axs[0].legend(loc="upper left", prop={"size": 20})
    axs[0].grid(True)
    axs[0].set_xlim(0, simTimesteps)
    axs[0].set_ylim(-1.5, 1.5)
    axs[0].set_title("Training Data", fontweight="bold", pad=10)

    # --- Middle subplot: simulations with online noise ---
    for y, c, s, m, lab in zip(trajectory_sims[0:3], colors, styles, markers, labels):
        axs[1].plot(np.arange(len(y)), y, linestyle=s,
                    marker=m, markersize=10, markevery=(0, 16),
                    markeredgecolor="black", color=c,
                    linewidth=3, label=lab)
    axs[1].set_ylabel(r"$y(t)$")
    axs[1].legend(loc="upper left", ncol=3, prop={"size": 20})
    axs[1].grid(True)
    axs[1].set_xlim(0, simTimesteps)
    axs[1].set_ylim(-1.5, 1.2)
    axs[1].set_title(fr"Proposed Controller With Online Noise $(\sigma^d,\sigma)=({offlineStd},{onlineStd})$",
                    fontweight="bold", pad=10)
    add_inset(axs[1], trajectory_sims[0:3], colors=colors, styles=styles, markers=markers)

    # --- Bottom subplot: simulations without online noise ---
    for y, c, s, m, lab in zip(trajectory_sims[3:6], colors, styles, markers, labels):
        axs[2].plot(np.arange(len(y)), y, linestyle=s,
                    marker=m, markersize=10, markevery=(0, 16),
                    markeredgecolor="black", color=c,
                    linewidth=3, label=lab)
    axs[2].set_xlabel(r"$t$")
    axs[2].set_ylabel(r"$y(t)$")
    axs[2].legend(loc="upper left", ncol=3, prop={"size": 20})
    axs[2].grid(True)
    axs[2].set_xlim(0, simTimesteps)
    axs[2].set_ylim(-1.5, 1.2)
    axs[2].set_title(fr"Proposed Controller Without Online Noise $(\sigma^d,\sigma)=({offlineStd},{0})$",
                    fontweight="bold", pad=10)
    add_inset(axs[2], trajectory_sims[3:6], colors=colors, styles=styles, markers=markers)

    plt.tight_layout()
    plt.savefig("output.pdf")
    plt.show()
    