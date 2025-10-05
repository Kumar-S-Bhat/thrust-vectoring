from src.aircraft.dynamics import Dynamics
import matplotlib.pyplot as plt
import numpy as np

# Initialize aircraft WITH actuator dynamics
aircraft = Dynamics(use_actuator_dynamics=True)

# Initial conditions (from trim solution)
state = np.array([
    200.0,  # u (m/s)
    5.0,    # w (m/s)
    0.0,    # q (rad/s)
    0.025,  # theta (rad)
    0.0,    # x (m)
    -5000.0  # z (m, negative for altitude)
])

# Simulation parameters
dt = 0.01  # 10ms time step
t_end = 10.0  # 10 second simulation
time = np.arange(0, t_end, dt)

# Storage arrays
states = np.zeros((len(time), 6))
alphas = np.zeros(len(time))

# Fixed controls (open-loop test)
controls = {
    'throttle': 0.4,
    'delta_p': np.radians(5)  # 5 degree nozzle deflection
}

# TIME INTEGRATION LOOP - This is where you use it
for i, t in enumerate(time):
    # Store current state
    states[i] = state

    # Calculate alpha for plotting
    u, w = state[0], state[1]
    alphas[i] = np.degrees(np.arctan2(w, u))

    # Get state derivatives from dynamics
    state_dot = aircraft.dynamics(state, controls, dt=dt)

    # Integrate (Euler method)
    state = state + state_dot * dt

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Velocity
axes[0].plot(time, states[:, 0], label='u (forward)')
axes[0].plot(time, states[:, 1], label='w (vertical)')
axes[0].set_ylabel('Velocity (m/s)')
axes[0].legend()
axes[0].grid(True)

# Pitch rate and attitude
axes[1].plot(time, np.degrees(states[:, 2]), label='q (pitch rate)')
axes[1].plot(time, np.degrees(states[:, 3]), label='theta (pitch angle)')
axes[1].set_ylabel('Angular (deg or deg/s)')
axes[1].legend()
axes[1].grid(True)

# Alpha
axes[2].plot(time, alphas)
axes[2].set_ylabel('Alpha (deg)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True)

plt.tight_layout()
# plt.savefig('flight_simulation.png')
plt.show()
