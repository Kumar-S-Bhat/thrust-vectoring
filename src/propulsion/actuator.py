import numpy as np


class NozzleActuator:
    def __init__(self, omega_n=40.0, max_deflection=20, max_rate=60):
        self.omega_n = omega_n
        self.max_deflection = np.radians(max_deflection)
        self.max_rate = np.radians(max_rate)
        self.delta_p = 0.0

    def reset(self, initial_position=0.0):
        self.delta_p = np.clip(
            initial_position, -self.max_deflection, self.max_deflection)

    def update(self, delta_cmd, dt):
        delta_cmd = np.clip(delta_cmd, -self.max_deflection,
                            self.max_deflection)
        delta_p_dot = self.omega_n*(delta_cmd-self.delta_p)
        delta_p_dot = np.clip(delta_p_dot, -self.max_rate, self.max_rate)

        self.delta_p += delta_p_dot*dt
        self.delta_p = np.clip(
            self.delta_p, -self.max_deflection, self.max_deflection)

        return self.delta_p

    def get_position(self):
        return self.delta_p

    def at_limit(self):
        return abs(self.delta_p) >= self.max_deflection*0.99
