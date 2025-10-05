import numpy as np


class NozzleActuator:
    """
    First-order actuator model for thrust vectoring nozzle.

    Dynamics: δ̇ = ω_n(δ_cmd - δ)

    Physical limits:
    - Position: ±20 degrees
    - Rate: 60 deg/s maximum
    """

    def __init__(self, omega_n=40.0, max_deflection=20.0, max_rate=60.0):
        """
        Initialize nozzle actuator.

        Args:
            omega_n (float): Natural frequency (rad/s), controls response speed
            max_deflection (float): Maximum nozzle angle (degrees)
            max_rate (float): Maximum rate of change (deg/s)
        """
        self.omega_n = omega_n  # rad/s, natural frequency

        # Convert limits to radians
        self.max_deflection = np.radians(max_deflection)  # rad
        self.max_rate = np.radians(max_rate)  # rad/s

        # Current state
        self.delta_p = 0.0  # Current nozzle position (rad)

    def reset(self, initial_position=0.0):
        """
        Reset actuator to initial position.

        Args:
            initial_position (float): Initial nozzle angle (rad)
        """
        self.delta_p = np.clip(initial_position,
                               -self.max_deflection,
                               self.max_deflection)

    def update(self, delta_p_cmd, dt):
        """
        Update actuator state with first-order dynamics and rate limiting.

        Args:
            delta_p_cmd (float): Commanded nozzle angle (rad)
            dt (float): Time step (s)

        Returns:
            float: Actual nozzle position (rad)
        """
        # Saturate command to physical limits
        delta_p_cmd = np.clip(delta_p_cmd,
                              -self.max_deflection,
                              self.max_deflection)

        # First-order dynamics: δ̇ = ω_n(δ_cmd - δ)
        delta_p_dot = self.omega_n * (delta_p_cmd - self.delta_p)

        # Apply rate limiting
        delta_p_dot = np.clip(delta_p_dot,
                              -self.max_rate,
                              self.max_rate)

        # Integrate with Euler method
        self.delta_p = self.delta_p + delta_p_dot * dt

        # Apply position limits again (safety)
        self.delta_p = np.clip(self.delta_p,
                               -self.max_deflection,
                               self.max_deflection)

        return self.delta_p

    def get_position(self):
        """Get current nozzle position (rad)."""
        return self.delta_p

    def at_limit(self):
        """
        Check if actuator is at position limit.

        Returns:
            bool: True if at ±max_deflection
        """
        return abs(self.delta_p) >= self.max_deflection * 0.99
