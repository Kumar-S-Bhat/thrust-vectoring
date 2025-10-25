import numpy as np
from src.control.pid import PID


class GainScheduledPID:
    def __init__(self, output_limits=(-100000000, 100000000)):
        self.output_limits = output_limits

        self.alpha_grid = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])

        # Kp: High at low alpha (need authority), decrease at high alpha (avoid oscillation)
        self.kp_schedule = np.array(
            [10000000, 9000000, 7000000, 4000000, 2000000, 1000000, 500000, 300000, 200000])

        # Ki: Moderate at low alpha, very low at high alpha (prevent windup in nonlinear regime)
        self.ki_schedule = np.array(
            [6000000, 5000000, 3000000, 1000000, 300000, 100000, 50000, 20000, 10000])

        # Kd: Increases with alpha to provide damping against oscillations
        self.kd_schedule = np.array(
            [300000, 400000, 600000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000])

        self.pid = PID(kp=self.kp_schedule[0],
                       ki=self.ki_schedule[0],
                       kd=self.kd_schedule[0],
                       output_limits=self.output_limits)

    def update(self, setpoint, measurement, alpha, dt):
        alpha_deg = np.degrees(alpha)
        alpha_deg = np.clip(alpha_deg, 0, 80)

        self.pid.kp = np.interp(alpha_deg, self.alpha_grid, self.kp_schedule)
        self.pid.ki = np.interp(alpha_deg, self.alpha_grid, self.ki_schedule)
        self.pid.kd = np.interp(alpha_deg, self.alpha_grid, self.kd_schedule)

        return self.pid.update(setpoint, measurement, dt)

    def reset(self):
        self.pid.reset()
