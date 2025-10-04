import numpy as np
from src.propulsion.actuator import NozzleActuator


class ThrustVectoringSystem:
    def __init__(self, l_arm=6.0):
        self.l_arm = l_arm
        self.actuator = NozzleActuator()

    def calculate_force(self, thrust, delta_p):
        Fx = thrust*np.cos(delta_p)
        Fz = thrust*np.sin(delta_p)

        M = Fz*self.l_arm

        return {
            'Fx': Fx,
            'Fz': Fz,
            'M': M,
        }

    def update(self, thrust, delta_cmd, dt):

        delta_p_actual = self.actuator.update(delta_cmd, dt)
        force_moment = self.calculate_force(thrust, delta_p_actual)

        return {
            **force_moment,
            'delta_p_actual': delta_p_actual,
            'delta_cmd': delta_cmd,
            'at_limit': self.actuator.at_limit()
        }
