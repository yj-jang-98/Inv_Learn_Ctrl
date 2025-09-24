import numpy as np
import torch
import gpytorch
import findIdx 

# PID Controller
def pid_controller(error, error_integral, error_derivative, Kp=1.0, Ki=0.1, Kd=0.05):
    return -(Kp * error + Ki * error_integral + Kd * error_derivative)


# Simulation with PID Controller
def simulate_trajectory_pid(y_init, u_init, sys, timesteps, dt, std=0.0, setpoint=0.0, Kp=1.0, Ki=0.1, Kd=0.05, rng=None):
    # --- rng setup ---
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    # --- Histories for outputs and input ---
    y_hist_true = [float(y_init[0]), float(y_init[1])]
    y_hist_meas = [
        y_hist_true[0] + rng.normal(0.0, std) if std > 0 else y_hist_true[0],
        y_hist_true[1] + rng.normal(0.0, std) if std > 0 else y_hist_true[1],
    ]
    u_hist = [float(u_init[0])]

    # --- Data storage --- 
    traj_true, traj_meas = [], []
    XI_meas, U = [], []

    # --- PID error accumulators ---
    error_integral = 0.0
    prev_error = y_hist_meas[0] - setpoint

    # --- Simulation loop ---
    for _ in range(timesteps - 2):
        y_k_true, y_km1_true = y_hist_true
        y_k_meas, y_km1_meas = y_hist_meas
        u_km1 = u_hist[0]

        # --- PID Controller ---
        error = y_k_meas - setpoint
        error_integral += error * dt
        error_derivative = (error - prev_error) / dt
        u_k = pid_controller(error, error_integral, error_derivative, Kp, Ki, Kd)

        # --- True system update ---
        y_kp1_true = sys(y_k_true, y_km1_true, u_km1, u_k)

        # --- Measurement noise ---
        y_kp1_meas = y_kp1_true + (rng.normal(0.0, std) if std > 0 else 0.0)

        # --- Store results ---
        XI_meas.append([y_kp1_meas, y_k_meas, y_km1_meas, u_km1])
        U.append(u_k)
        traj_true.append(y_kp1_true)
        traj_meas.append(y_kp1_meas)

        # --- Roll histories for next step ---
        y_hist_true = [y_kp1_true, y_k_true]
        y_hist_meas = [y_kp1_meas, y_k_meas]
        u_hist = [u_k]
        prev_error = error

    return (
        np.array(traj_true),
        np.array(traj_meas),
        np.array(XI_meas),
        np.array(U)
    )


# Simulation with proposed controller
def simulate_trajectory(sys, curr_state, A_all, XI, ZETA, model, likelihood, timesteps, std, rng=None):
    # --- rng setup ---
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    trajectory_sim = []
    states_visited = [curr_state.copy()]  # store initial state

    # --- Initial true and noisy state ---
    y_k, y_km1, u_km1 = curr_state
    y_k_meas = y_k + rng.normal(0.0, std)
    y_km1_meas = y_km1 + rng.normal(0.0, std)
    curr_state_meas = [y_k_meas, y_km1_meas, u_km1]

    for k in range(timesteps - 1):
        # --- Find matching index and set the reference point ---
        idx_final = None
        for delta, A in A_all.items():
            matched, idx_match = findIdx.findIdx(curr_state_meas, ZETA, XI, A)
            if matched:
                idx_final = idx_match
                break
        # --- If not found, terminate
        if idx_final is None:
            print(f"[simulate_trajectory] No matching index at step {k}. Stopping early.")
            return np.array(trajectory_sim), np.array(states_visited)

        # --- Compute control input ---
        yplus = XI[idx_final, 0]
        inv_input = np.concatenate(([yplus], curr_state_meas)).reshape(1, -1)
        test_x = torch.from_numpy(inv_input).float()

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(test_x))
            u_k = pred_dist.mean.item()

        # --- Compute next output ---
        y_kp1 = sys(y_k, y_km1, u_k, u_km1)
        trajectory_sim.append(y_kp1)

        # --- Add measurement noise ---
        y_kp1_meas = y_kp1 + rng.normal(0.0, std)

        # --- Roll forward ---
        y_k, y_km1, u_km1 = y_kp1, y_k, u_k
        curr_state = np.array([y_k, y_km1, u_km1])
        curr_state_meas = [y_kp1_meas, curr_state_meas[0], u_km1]

        states_visited.append(curr_state.copy())

    return np.array(trajectory_sim), np.array(states_visited)