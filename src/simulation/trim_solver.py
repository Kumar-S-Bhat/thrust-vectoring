import numpy as np
from src.aircraft.dynamics import Dynamics
from scipy.optimize import least_squares


class Trim_solver():
    def __init__(self):
        self.aircraft = Dynamics()

    def trim_cost_function(self, control_vars, target_conditions):
        """function to be optimized

        Args:
        control_vars = [throttle,delta_p,alpha]
        target_condition = {velocity,flight_path_angle,altitude}

        Returns:
        Residual array
        [u_dot, w_dot, q_dot, gamma_dot]"""

        throttle, delta_p, alpha = control_vars

        V = target_conditions.get('velocity')
        h = target_conditions.get('altitude')
        gamma = target_conditions.get('flight_path_angle')

        u = V*np.cos(alpha)
        w = V*np.sin(alpha)
        theta = gamma + alpha

        x = 0
        z = -h
        q = 0      # At trim q = 0

        state = [u, w, q, theta, x, z]
        controls = {'throttle': throttle, 'delta_p': delta_p}
        state_dot = self.aircraft.dynamics(state, controls)

        u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot = state_dot

        # Division by 0 risk
        if V > 1e-12:
            alpha_dot = (u*w_dot-w*u_dot)/V**2
        else:
            alpha_dot = 0

        gamma_dot = theta_dot - alpha_dot

        return [u_dot, w_dot, q_dot, gamma_dot]

    def find_trim(self, velocity, flight_path_angle=0, altitude=1000):
        """
        Aim: Find trimmed condition
        """

        target_conditions = {
            'velocity': velocity,
            'flight_path_angle': flight_path_angle,
            'altitude': altitude
        }

        initial_controls = [0.5, 0, 0.08]

        result = least_squares(
            fun=self.trim_cost_function,
            x0=initial_controls,
            args=(target_conditions,),
            bounds=([0, -np.radians(20), -np.radians(10)],
                    [1, np.radians(20), np.radians(20)])
        )

        if result.success:
            throttle, delta_p, alpha = result.x
            residuals = result.fun
            max_residual = np.max(np.abs(residuals))

            if max_residual < 1e-3:

                return {
                    'target_conditions': target_conditions,
                    'success': True,
                    'throttle': throttle,
                    'delta_p': delta_p,
                    'alpha': alpha,
                    'residuals': residuals,
                    'max_residual': max_residual,
                }
            else:
                return {
                    'target_conditions': target_conditions,
                    'success': False,
                    'throttle': throttle,
                    'delta_p': delta_p,
                    'alpha': alpha,
                    'residuals': residuals,
                    'message': f'Poor trim quality: max residual {max_residual:.2e}',
                    'max_residual': max_residual
                }
