import numpy as np
from src.aircraft.dynamics import Dynamics
from scipy.optimize import least_squares


class Trim_solver():
    """
    Trim solver with elevator deflection.
    Solves: [throttle, delta_p, delta_e, alpha] for zero accelerations.
    """

    def __init__(self):
        self.aircraft = Dynamics(use_actuator_dynamics=False)

    def trim_cost_function(self, control_vars, target_conditions):
        """function to be optimized

        Args:
        control_vars = [throttle,delta_p,delta_e,alpha]
        target_condition = {velocity,flight_path_angle,altitude}

        Returns:
        Residual array
        [u_dot, w_dot, q_dot, gamma_dot]"""

        throttle, delta_p, delta_e, alpha = control_vars

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
        controls = {'throttle': throttle,
                    'delta_p': delta_p, 'delta_e': delta_e}
        state_dot = self.aircraft.dynamics(state, controls)

        u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot = state_dot

        # Division by 0 risk
        if V > 1e-12:
            alpha_dot = (u*w_dot-w*u_dot)/V**2
        else:
            alpha_dot = 0.0

        gamma_dot = theta_dot - alpha_dot

        return [u_dot, w_dot, q_dot, gamma_dot]

    def find_trim(self, velocity, flight_path_angle=0.0, altitude=1000.0,
                  alpha_guess=None, initial_controls=None):
        """
        Find trimmed flight condition.

        Args:
            velocity (float): Target airspeed (m/s)
            flight_path_angle (float): Target gamma (rad)
            altitude (float): Target altitude (m)
            alpha_guess (float): Initial guess for alpha (rad), optional
            initial_controls (list): [throttle, delta_p, delta_e, alpha], optional

        Returns:
            dict: Trim solution
        """
        target_conditions = {
            'velocity': velocity,
            'flight_path_angle': flight_path_angle,
            'altitude': altitude
        }

        # Estimate initial alpha if not provided
        if alpha_guess is None:
            rho = self.aircraft.engine.atmosphere(altitude)
            q_bar = 0.5 * rho * velocity**2
            W = self.aircraft.m * self.aircraft.g
            CL_required = W / (q_bar * self.aircraft.S)

            # Rough linear estimate (works for low alpha)
            alpha_guess = np.clip(
                CL_required / 5.5, np.radians(-5), np.radians(15))

        # Initial guess for controls
        if initial_controls is None:
            throttle_guess = 0.3
            delta_p_guess = 0.0  # Start with no TVC
            delta_e_guess = 0.0  # Start with neutral elevator
            initial_controls = [throttle_guess,
                                delta_p_guess, delta_e_guess, alpha_guess]

        result = least_squares(
            fun=self.trim_cost_function,
            x0=initial_controls,
            args=(target_conditions,),
            bounds=([0.0, -np.radians(20), -np.radians(25), -np.radians(10)],
                    [1.0, np.radians(20), np.radians(25), np.radians(20)]),
            ftol=1e-9,    # Tight: forces solver to work hard
            xtol=1e-9,    # Tight: ensures precise parameter convergence
            max_nfev=2000  # Enough iterations
        )

        if result.success:
            throttle, delta_p, delta_e, alpha = result.x
            residuals = result.fun
            max_residual = np.max(np.abs(residuals))

            if max_residual < 1e-3:
                # Good trim: residuals small enough
                return {
                    'success': True,
                    'throttle': throttle,
                    'delta_p': delta_p,
                    'delta_e': delta_e,
                    'alpha': alpha,
                    'residuals': residuals,
                    'max_residual': max_residual,
                    'target_conditions': target_conditions,
                }
            else:
                # Solver converged, but trim quality poor
                return {
                    'success': False,
                    'throttle': throttle,
                    'delta_p': delta_p,
                    'delta_e': delta_e,
                    'alpha': alpha,
                    'residuals': residuals,
                    'message': f'Poor trim quality: max residual {max_residual:.2e}',
                    'max_residual': max_residual,
                    'target_conditions': target_conditions
                }
        else:
            # Solver didn't converge at all
            return {
                'success': False,
                'throttle': throttle,
                'delta_p': delta_p,
                'delta_e': delta_e,
                'alpha': alpha,
                'residuals': residuals,
                'message': f'Solver failed to converge: {result.message}',
                'max_residual': max_residual,
                'target_conditions': target_conditions
            }
