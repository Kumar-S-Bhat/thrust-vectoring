"""
Validate PID response for nozzle position control.
Used to check performance of pre-tuned gains.
"""

import numpy as np
from src.control.pid import PID
import matplotlib.pyplot as plt


print("Testing PID Controller")

# Use optimal gains found via grid search in examples/tune_pid_nozzle.py
kp = 0.9
ki = 40.0
kd = 0.0

pid = PID(
    kp=kp,
    ki=ki,
    kd=kd,
    output_limits=(-np.radians(20), np.radians(20)),
    derivative_filter_tau=0.01  # 10ms filter
)

# Simple first-order plant model (like actuator)
omega_n = 40.0  # rad/s
measurement = 0.0

# Simulation
dt = 0.001  # 1ms
t_end = 1.0
time = np.arange(0, t_end, dt)

# Step command at t=0.1s
setpoint = np.where(time < 0.1, 0.0, np.radians(15.0))

# Storage
output = np.zeros_like(time)     # stores local setpoint
response = np.zeros_like(time)   # stores delta_p_actual
p_terms = np.zeros_like(time)
i_terms = np.zeros_like(time)
d_terms = np.zeros_like(time)

# Simulate closed loop
pid.reset()
for i, t in enumerate(time):
    # PID output
    control = pid.update(setpoint[i], measurement, dt)
    output[i] = control

    # Get individual terms for plotting
    terms = pid.get_terms(setpoint[i], measurement)
    p_terms[i] = terms['p']
    i_terms[i] = terms['i']
    d_terms[i] = terms['d']

    # Plant dynamics: ẋ = ω(u - x)
    measurement_dot = omega_n * (control - measurement)
    measurement += measurement_dot * dt
    response[i] = measurement

# Calculate metrics
setpoint_final = setpoint[-1]   # total required nozzle deflection: here 15°
final_value = response[-1]      # delta_p_actual after t_end seconds

# Rise time (10%-90%)
idx_10 = np.where(response >= 0.1 * setpoint_final)[0]
idx_90 = np.where(response >= 0.9 * setpoint_final)[0]
if len(idx_10) > 0 and len(idx_90) > 0:
    rise_time = time[idx_90[0]] - time[idx_10[0]]
else:
    rise_time = np.nan

# Overshoot
overshoot = (np.max(response) - setpoint_final) / setpoint_final * 100

# Settling time (2%)
settling_band = 0.02 * setpoint_final
settled = np.where(np.abs(response - setpoint_final) <= settling_band)[0]
settle_time = time[settled[0]] if len(settled) > 0 else t_end

print(f"kp = {kp}, ki = {ki}, kd = {kd}")
print(f"\nStep Response Metrics (0 → 15°):")
print(f"  Rise time:           {rise_time:.3f} s")
print(f"  Overshoot:           {overshoot:.1f}%")
print(f"  Settling time (2%):  {settle_time:.3f} s")
print(
    f"  Steady-state error:  {np.degrees(setpoint_final - final_value):.3f}°")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Response
ax1 = axes[0]
ax1.plot(time, np.degrees(setpoint), 'r--', label='Setpoint', linewidth=2)
ax1.plot(time, np.degrees(response), 'b-', label='Response', linewidth=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Position (deg)')
ax1.set_title('PID Step Response')
ax1.legend()

# Control output
ax2 = axes[1]
ax2.plot(time, np.degrees(output), 'g-', linewidth=2)
ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Limits')
ax2.axhline(y=-20, color='r', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Control Output (deg)')
ax2.legend()

# PID terms
ax3 = axes[2]
ax3.plot(time, np.degrees(p_terms), label='P term', linewidth=1.5)
ax3.plot(time, np.degrees(i_terms), label='I term', linewidth=1.5)
ax3.plot(time, np.degrees(d_terms), label='D term', linewidth=1.5)
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('PID Terms (deg)')
ax3.set_xlabel('Time (s)')
ax3.legend()

plt.tight_layout()
# plt.savefig('pid_response.png', dpi=150)
# print(f"\nPlot saved as 'pid_response.png'")
plt.show()

# Check performance
print("\nPerformance Assessment:")
if settle_time < 0.5 and abs(overshoot) < 10:
    print("  Good performance!")
elif settle_time < 0.5:
    print("  ⚠️  Fast but too much overshoot")
elif abs(overshoot) < 10:
    print("  ⚠️  Well-damped but slow")
else:
    print("  ❌ Poor performance - tune gains")
