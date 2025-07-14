import numpy as np
import torch
import gpytorch
from findIdx import *

# --- PID Controller ---
def pid_controller(error, error_integral, error_derivative, Kp=1.0, Ki=0.1, Kd=0.05):
    u_k = - (Kp * error + Ki * error_integral + Kd * error_derivative)
    return u_k

# --- Simulation Function with PID Controller ---
def simulate_trajectory_pid(y_init, u_init, sys, timesteps, dt, noise=0.0, setpoint=0.0, Kp=1.0, Ki=0.1, Kd=0.05):
    y_history = y_init.copy()
    u_history = u_init.copy()

    traj = []
    XI = []  # [y(t+1), y(t), y(t-1), u(t-1)]
    U = []   # Control input u(t)

    error_integral = 0.0
    prev_error = y_history[0] - setpoint

    for _ in range(timesteps - 2):
        y_k, y_km1 = y_history[0], y_history[1]
        u_km1 = u_history[0]

        # Compute errors
        error = y_k - setpoint
        error_integral += error * dt
        error_derivative = (error - prev_error) / dt

        # PID control input
        u_k = pid_controller(error, error_integral, error_derivative, Kp, Ki, Kd)

        # Add noise
        u_k += np.random.normal(0, noise)

        # Apply system dynamics
        y_kp1 = sys(y_k, y_km1, u_km1, u_k)

        # Store data
        XI.append([y_kp1, y_k, y_km1, u_km1])
        U.append(u_k)
        traj.append(y_kp1)

        # Update histories
        y_history = [y_kp1, y_k]
        u_history = [u_k]
        prev_error = error

    return np.array(traj), np.array(XI), np.array(U)

def simulate_trajectory(sys, curr_state, A_all, XI, ZETA, model, likelihood, timesteps):
    trajectory_sim = []
    states_visited = [curr_state.copy()]  # <-- store initial state

    y_k, y_km1, u_km1 = curr_state[0], curr_state[1], curr_state[2]

    for k in range(timesteps - 1):
        matched = False
        idx_final = None
        
        for delta, A in A_all.items():
            matched, idx_match = findIdx(curr_state, ZETA, XI, A)
            if matched:
                idx_final = idx_match
                break
        
        # Predict next control input using inverse GP model
        yplus = XI[idx_final, 0]
        inv_input = np.concatenate(([yplus], curr_state)).reshape(1, -1)
        test_x = torch.from_numpy(inv_input).float()

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(test_x))
            u_k = pred_dist.mean.item()

        # Simulate next output
        y_kp1 = sys(y_k, y_km1, u_k, u_km1)
        trajectory_sim.append(y_kp1)

        # Update for next step
        y_km1 = y_k
        y_k = y_kp1
        u_km1 = u_k
        curr_state = np.array([y_k, y_km1, u_km1])
        
        states_visited.append(curr_state.copy())  # <-- store updated state
    return np.array(trajectory_sim), np.array(states_visited)