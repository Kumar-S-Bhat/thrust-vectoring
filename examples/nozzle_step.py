import numpy as np
import matplotlib.pyplot as plt
from src.aircraft.dynamics import Dynamics
from src.simulation.trim_solver import Trim_solver
# Test the actuator response:

velocity = 200
altitude = 5000

# Get trim condition first
trim_solver = Trim_solver()
trim_result = trim_solver.find_trim(
    velocity=velocity,
    flight_path_angle=0,
    altitude=altitude
)

# Initialize aircraft WITH actuator dynamics
aircraft = Dynamics(use_actuator_dynamics=True)

# Start from trim state
u = velocity * np.cos(trim_result['alpha'])
w = velocity * np.sin(trim_result['alpha'])
state = np.array([u, w, 0, trim_result['alpha'], 0, -altitude])

# Simulation
dt = 0.01
t_end = 2.0
time = np.arange(0, t_end, dt)

nozzle_commands = np.zeros(len(time))
nozzle_actual = np.zeros(len(time))

# Step command at t=0.5s: 0 -> 10 degrees
for i, t in enumerate(time):
    if t < 0.5:
        delta_cmd = 0.0
    else:
        delta_cmd = np.radians(10)

    nozzle_commands[i] = np.degrees(delta_cmd)

    controls = {
        'throttle': trim_result['throttle'],
        'delta_p': delta_cmd
    }

    state_dot = aircraft.dynamics(state, controls, dt=dt)
    state = state + state_dot * dt

    # Get actual nozzle position from actuator
    nozzle_actual[i] = np.degrees(aircraft.tvc.actuator.get_position())

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, nozzle_commands, 'r--', label='Command', linewidth=2)
plt.plot(time, nozzle_actual, 'b-', label='Actual', linewidth=2)
plt.axhline(y=20, color='k', linestyle=':', label='Limits')
plt.axhline(y=-20, color='k', linestyle=':')
plt.xlabel('Time (s)')
plt.ylabel('Nozzle Angle (deg)')
plt.title('Nozzle Actuator Step Response')
plt.legend()
plt.grid(True)
# plt.savefig('nozzle_step.png')
plt.show()
