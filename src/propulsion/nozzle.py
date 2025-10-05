import numpy as np
from src.propulsion.actuator import NozzleActuator


class ThrustVectoringSystem:
    """
    Complete thrust vectoring system combining thrust model and nozzle.
    """

    def __init__(self, l_arm=6.0):
        """
        Initialize TVC system.

        Args:
            l_arm (float): Moment arm from CG to nozzle (m)
        """
        self.l_arm = l_arm

        # Create actuator
        self.actuator = NozzleActuator()

    def calculate_forces_moments(self, thrust, delta_p):
        """
        Calculate forces and moments from thrust vectoring.

        Args:
            thrust (float): Thrust magnitude (N)
            delta_p (float): Nozzle deflection angle (rad)

        Returns:
            dict: {'Fx': forward force, 'Fz': vertical force, 'My': pitch moment}
        """
        # Force components in body frame
        Fx = thrust * np.cos(delta_p)  # Forward component
        Fz = thrust * np.sin(delta_p)  # Vertical component

        # Pitch moment = vertical force × moment arm
        My = Fz * self.l_arm

        return {
            'Fx': Fx,
            'Fz': Fz,
            'My': My
        }

    def update(self, thrust, delta_p_cmd, dt):
        """
        Update TVC system for one time step.

        Args:
            thrust (float): Current thrust (N)
            delta_p_cmd (float): Commanded nozzle angle (rad)
            dt (float): Time step (s)

        Returns:
            dict: Forces, moments, and actuator state
        """
        # Update actuator dynamics
        delta_p_actual = self.actuator.update(delta_p_cmd, dt)

        # Calculate resulting forces and moments
        forces_moments = self.calculate_forces_moments(thrust, delta_p_actual)

        return {
            **forces_moments,
            'delta_p_actual': delta_p_actual,
            'delta_p_cmd': delta_p_cmd,
            'at_limit': self.actuator.at_limit()
        }
