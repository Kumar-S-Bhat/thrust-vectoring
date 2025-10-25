import numpy as np
from src.aircraft.aerodynamics import AeroTable
from src.propulsion.thrust_model import ThrustModel
from src.propulsion.nozzle import ThrustVectoringSystem


class Dynamics:
    """
    Simplified longitudinal aircraft dynamics for thrust vectoring study.

    State vector: [u, w, q, theta, x, z]
    - u, w: body-frame velocities (m/s)
    - q: pitch rate (rad/s)  
    - theta: pitch angle (rad)
    - x, z: position in NED frame (m)
    """

    def __init__(self, use_actuator_dynamics=True):
        # Aircraft parameters (F-16 class)
        self.m = 9300.0              # mass
        self.Iy = 55814.0           # pitch moment of inertia
        self.S = 27.87              # wing reference area
        self.chord = 3.45           # mean aerodynamic chord
        self.g = 9.81
        self.l_arm = 6.0            # nozzle moment arm from CG

        # Store flag
        self.use_actuator_dynamics = use_actuator_dynamics

        # Initialize subsystems
        self.aero = AeroTable('src/data/aero_tables.csv')
        self.engine = ThrustModel(thrust_sl=130000.0, rho_sl=1.225)
        self.tvc = ThrustVectoringSystem(l_arm=self.l_arm)

        self.CM_q = -15.0           # pitch damping coefficient

    def dynamics(self, state, controls=None, dt=0.01):
        """
        Longitudinal aircraft dynamics equations

        Args:
            state (array): [u, w, q, theta, x, z]
            controls (dict): {'throttle': 0-1, 'delta_p': rad, 'delta_e': rad} 
                           If None, uses throttle=0.5, delta_p=0, delta_e=0.0

        Returns:
            array: State derivatives [u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot]
        """
        # Unpack state
        u, w, q, theta, x, z = state

        # Handle controls with defaults
        if controls is None:
            throttle = 0.5
            delta_p_cmd = 0.0  # nozzle pitch
            delta_e = 0.0
        else:
            throttle = controls.get('throttle', 0.5)
            delta_p_cmd = controls.get('delta_p', 0.0)
            delta_e = controls.get('delta_e')

        # Current altitude (positive up, z is negative in NED)
        h = -z

        # Calculate flight conditions
        V = np.sqrt(u**2 + w**2)
        if V < 1e-6:  # Avoid division by zero
            V = 1e-6

        alpha = np.arctan2(w, u)  # Angle of attack
        rho = self.engine.atmosphere(h)
        q_infty = 0.5 * rho * V**2  # Dynamic pressure

        # Get thrust
        T = self.engine.thrust_force(h, throttle)

        # Aerodynamic coefficients
        CL, CD, CM_static, Cm_de = self.aero.get_coefficients(alpha)
        CM = CM_static + self.CM_q * \
            (q * self.chord) / (2 * V) - Cm_de * delta_e

        # Aerodynamic forces and moments (body frame)
        Fx_aero = -q_infty * self.S * CD  # Drag (opposes motion)
        Fz_aero = -q_infty * self.S * CL  # Lift (negative in body z)
        M_aero = q_infty * self.S * self.chord * CM  # Pitch moment

        # Get thrust vectoring forces/moments
        if self.use_actuator_dynamics:
            tvc_output = self.tvc.update(T, delta_p_cmd, dt)
            Fx_thrust = tvc_output['Fx']
            Fz_thrust = tvc_output['Fz']
            M_thrust = tvc_output['My']
        else:
            # Instant response for trim calculations
            forces = self.tvc.calculate_forces_moments(T, delta_p_cmd)
            Fx_thrust = forces['Fx']
            Fz_thrust = forces['Fz']
            M_thrust = forces['My']

        # Total forces and moments
        Fx_total = Fx_aero + Fx_thrust
        Fz_total = Fz_aero + Fz_thrust
        M_total = M_aero + M_thrust

        # Standard aircraft equations: m(u̇ + qw - rv) = ΣFx - mg sin θ
        #                              m(ẇ - qu + pv) = ΣFz + mg cos θ
        # For longitudinal motion: r = p = v = 0
        u_dot = (Fx_total - self.m * self.g * np.sin(theta)) / self.m - q * w
        w_dot = (Fz_total + self.m * self.g * np.cos(theta)) / self.m + q * u

        # Moment equation
        q_dot = M_total / self.Iy

        # === HARD-CLAMP PITCH ACCELERATION ===
        MAX_PITCH_ACCEL_RAD = np.radians(5000)
        q_dot = np.clip(q_dot, -MAX_PITCH_ACCEL_RAD, MAX_PITCH_ACCEL_RAD)

        # Attitude kinematics
        theta_dot = q  # For longitudinal motion

        # Position kinematics (NED frame)
        x_dot = u * np.cos(theta) + w * np.sin(theta)
        # Negative because NED z down
        z_dot = -u * np.sin(theta) + w * np.cos(theta)

        return np.array([u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot])

    def get_flight_path_angle(self, u, w, theta):
        """
        Calculate flight path angle

        Args:
            u, w (float): Body velocities (m/s)
            theta (float): Pitch angle (rad)

        Returns:
            float: Flight path angle gamma (rad)
        """
        alpha = np.arctan2(w, u)
        gamma = theta - alpha
        return gamma

    def get_airspeed_alpha(self, u, w):
        """
        Calculate airspeed and angle of attack

        Args:
            u, w (float): Body velocities (m/s)

        Returns:
            tuple: (airspeed, alpha) in (m/s, rad)
        """
        V = np.sqrt(u**2 + w**2)
        alpha = np.arctan2(w, u)
        return V, alpha
