"""
PID Gain Optimization via Grid Search

Performs brute-force parameter sweep to find optimal
PID gains for thrust vectoring nozzle actuator.

Generates:
- Performance heatmap (kp vs ki at best kd)
- Best step response plot

Used to justify gains used in src/control/pid.py
"""


import numpy as np
import matplotlib.pyplot as plt

# Parameter ranges
kp_range = np.linspace(-2, 4, 50)
ki_range = np.linspace(0, 40, 50)
kd_range = np.linspace(0, 2, 30)

# Create all combinations
kp_grid, ki_grid, kd_grid = np.meshgrid(
    kp_range, ki_range, kd_range, indexing='ij')
kp_flat = kp_grid.flatten()
ki_flat = ki_grid.flatten()
kd_flat = kd_grid.flatten()
n = len(kp_flat)

# Simulation setup
dt = 0.001
time = np.arange(0, 1.0, dt)
setpoint = np.where(time < 0.1, 0.0, np.radians(15.0))
setpoint_final = setpoint[-1]

# Plant parameters
omega_n = 40.0
limits = (-np.radians(20), np.radians(20))

# State arrays
measurement = np.zeros((n, len(time)))
integrals = np.zeros(n)
prev_err = np.zeros(n)
prev_deriv = np.zeros(n)

# Filter
tau = 0.01
alpha = dt / (tau + dt)

# Simulate
for i in range(len(time)):
    """
    Avoiding function call overhead by implementing PID class logic here.
    """
    err = setpoint[i] - measurement[:, i]

    p = kp_flat * err
    integrals += ki_flat * err * dt
    raw_d = (err - prev_err) / dt
    filt_d = alpha * raw_d + (1 - alpha) * prev_deriv
    d = kd_flat * filt_d

    u = np.clip(p + integrals + d, limits[0], limits[1])

    if i < len(time) - 1:
        measurement[:, i + 1] = measurement[:, i] + \
            omega_n * (u - measurement[:, i]) * dt

    prev_err = err
    prev_deriv = filt_d

# Calculate metrics
rise_time = np.full(n, np.nan)
overshoot = np.zeros(n)
settle = np.full(n, 1.0)

for idx in range(n):
    r = measurement[idx, :]

    # Rise time
    i10 = np.where(r >= 0.1 * setpoint_final)[0]
    i90 = np.where(r >= 0.9 * setpoint_final)[0]
    if len(i10) > 0 and len(i90) > 0:
        rise_time[idx] = time[i90[0]] - time[i10[0]]

    # Overshoot
    overshoot[idx] = (np.max(r) - setpoint_final) / setpoint_final * 100

    # Settling time (2%)
    settled = np.where(np.abs(r - setpoint_final) <= 0.02 * setpoint_final)[0]
    if len(settled) > 0:
        settle[idx] = time[settled[0]]

# Score and find best
# Evaluate performance using a weighted cost function
# Prioritizes low settling time, zero steady-state error, minimal overshoot
# Best configuration minimizes the overall score
ss_err = np.degrees(np.abs(setpoint_final - measurement[:, -1]))
score = 2*settle + 0.5*np.abs(overshoot) + 5*ss_err + 0.1*rise_time
best = np.nanargmin(score)

print(
    f"Best: kp={kp_flat[best]:.3f}, ki={ki_flat[best]:.3f}, kd={kd_flat[best]:.3f}")
print(
    f"Rise: {rise_time[best]:.4f}s, Over: {overshoot[best]:.2f}%, Settle: {settle[best]:.4f}s")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(time, np.degrees(setpoint), 'k--', linewidth=2)
ax[0].plot(time, np.degrees(measurement[best, :]), 'b-', linewidth=2)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Angle (deg)')
ax[0].grid(True, alpha=0.3)
ax[0].set_title(
    f'Best Response\nkp={kp_flat[best]:.3f}, ki={ki_flat[best]:.3f}, kd={kd_flat[best]:.3f}')

# Score heatmap (kp vs ki, at best kd)
scores_3d = score.reshape(kp_grid.shape)
best_kd_idx = np.argmin(np.abs(kd_range - kd_flat[best]))
im = ax[1].imshow(scores_3d[:, :, best_kd_idx].T, origin='lower', aspect='auto',
                  extent=[kp_range[0], kp_range[-1],
                          ki_range[0], ki_range[-1]],
                  cmap='viridis_r')
ax[1].plot(kp_flat[best], ki_flat[best], 'r*', markersize=10)
ax[1].set_xlabel('kp')
ax[1].set_ylabel('ki')
ax[1].set_title(f'Score Map (kd={kd_flat[best]:.2f})\n ★ = Best')
plt.colorbar(im, ax=ax[1], label='Score (lower is better)')

plt.tight_layout()
plt.show()
