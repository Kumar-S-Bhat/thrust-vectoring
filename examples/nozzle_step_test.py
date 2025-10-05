import numpy as np
from src.propulsion.actuator import NozzleActuator
import matplotlib.pyplot as plt

print("Testing Nozzle Actuator Dynamics")

# Create actuator
actuator = NozzleActuator(omega_n=40.0)

# Simulation parameters
dt = 0.001  # 1ms time step
t_end = 1.0  # 1 second simulation
time = np.arange(0, t_end, dt)

# Command: step from 0 to 15 degrees at t=0.1s
nozzle_commands = np.where(time < 0.1, 0.0, 15.0)
delta_cmd = np.radians(nozzle_commands)

# Storage arrays
nozzle_actual = np.zeros_like(time)

# Simulate
actuator.reset()
for i, t in enumerate(time):
    pos = actuator.update(delta_cmd[i], dt)
    nozzle_actual[i] = np.degrees(pos)

# Calculate metrics
final_value = nozzle_actual[-1]
rise_time_idx = np.where(nozzle_actual >= 0.9 * final_value)[0][0]
rise_time = time[rise_time_idx]

overshoot = (np.max(nozzle_actual) - final_value) / final_value * 100

settling_band = 0.02 * final_value
settled_idx = np.where(np.abs(nozzle_actual - final_value) <= settling_band)[0]
if len(settled_idx) > 0:
    settle_time = time[settled_idx[0]]
else:
    settle_time = t_end

print(f"\nStep Response Metrics (0 → 15°):")
print(f"  Rise time (10%-90%): {rise_time:.3f} s")
print(f"  Overshoot:           {overshoot:.1f}%")
print(f"  Settling time (2%):  {settle_time:.3f} s")
print(f"  Final value:         {final_value:.2f}°")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# nozzle_actual
ax1.plot(time, nozzle_commands, 'r--', label='Command', linewidth=2)
ax1.plot(time, nozzle_actual, 'b-', label='Actual', linewidth=2)
ax1.axhline(y=20, color='k', linestyle=':', alpha=0.5, label='Limits')
ax1.axhline(y=-20, color='k', linestyle=':', alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Nozzle Angle (deg)')
ax1.set_title('Nozzle Actuator Step Response')
ax1.legend()
ax1.set_xlim([0, t_end])

# Rate
rate = np.gradient(nozzle_actual, dt)
ax2.plot(time, rate, 'g-', linewidth=2)
ax2.axhline(y=60, color='r', linestyle='--', label='Rate Limit')
ax2.axhline(y=-60, color='r', linestyle='--')
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Nozzle Rate (deg/s)')
ax2.set_xlabel('Time (s)')
ax2.legend()
ax2.set_xlim([0, t_end])

plt.tight_layout()
# plt.savefig('nozzle_step_response')
plt.show()

# Test rate limiting
print("\n" + "="*60)
print("Testing Rate Limiting")

# Large step command
actuator.reset()
time_fast = np.arange(0, 0.5, dt)
command_fast = np.radians(20.0)  # Full deflection command

nozzle_actual_fast = np.zeros_like(time_fast)
for i, t in enumerate(time_fast):
    pos = actuator.update(command_fast, dt)
    nozzle_actual_fast[i] = np.degrees(pos)

max_rate_observed = np.max(np.abs(np.gradient(nozzle_actual_fast, dt)))

print(f"  Commanded:     20° instantly")
print(f"  Max rate:      {max_rate_observed:.1f} deg/s")
print(f"  Rate limit:    60 deg/s")
print(
    f"  Time to reach: {time_fast[np.argmax(nozzle_actual_fast >= 19.9)]:.3f} s")

if max_rate_observed <= 60.5:  # Small tolerance for numerical errors
    print(f"  ✅ Rate limiting working correctly")
else:
    print(f"  ❌ Rate limit violated!")
