from src.aircraft.dynamics import Dynamics


# Create aircraft
aircraft = [0, 0]
aircraft[0] = Dynamics(use_actuator_dynamics=True)
aircraft[1] = Dynamics(use_actuator_dynamics=False)

# Initial state: 100 m/s forward, small climb, 1000m altitude
state = [100.0, 5.0, 0.0, 0.05, 0.0, -1000.0]  # u, w, q, theta, x, z

# Controls
controls = {'throttle': 0.7, 'delta_p': 0.1, 'delta_e': 0.2}

# Calculate derivatives
for i in range(2):
    state_dot = aircraft[i].dynamics(state, controls)

    print("\nState derivatives:")
    print(f"u_dot = {state_dot[0]:.3f} m/s²")
    print(f"w_dot = {state_dot[1]:.3f} m/s²")
    print(f"q_dot = {state_dot[2]:.3f} rad/s²")
    print(f"theta_dot = {state_dot[3]:.3f} rad/s")
    print(f"x_dot = {state_dot[4]:.3f} m/s")
    print(f"z_dot = {state_dot[5]:.3f} m/s")
