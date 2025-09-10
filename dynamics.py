import numpy as np


class Dynamics:
    """
    Simplified longitudinal aircraft dynamics for thrust vectoring study.

    State vector: [u, w, q, theta, x, z]
    - u, w: body-frame velocities (m/s)
    - q: pitch rate (rad/s)  
    - theta: pitch angle (rad)
    - x, z: position in NED frame (m)
    """

    def __init__(self):
        # Aircraft parameters (F-16 class)
        self.m = 9300.0              # kg, mass
        self.Iy = 55814.0           # kg*m^2, pitch moment of inertia
        self.S = 27.87              # m^2, wing reference area
        self.chord = 3.45           # m, mean aerodynamic chord
        self.g = 9.81               # m/s^2, gravity

        # Engine/nozzle geometry
        self.l_arm = 4.0            # m, nozzle moment arm from CG

        # Atmospheric model (ISA)
        self.rho_sl = 1.225         # kg/m^3, sea level density
        self.Temp_sl = 288.15       # K, sea level temperature
        self.L = 0.0065             # K/m, temperature lapse rate

        # Propulsion
        self.thrust_sl = 130000.0   # N, sea level static thrust

        # Simplified aerodynamic coefficients (will use tables later)
        self.CL_alpha = 5.5         # per radian, lift curve slope
        self.CD_0 = 0.025           # zero-lift drag coefficient
        self.CD_alpha = 0.3         # drag due to alpha squared
        self.CM_alpha = -0.8        # pitch moment curve slope (static margin)
        self.CM_q = -15.0           # pitch damping coefficient

    def atmosphere(self, altitude):
        """
        ISA atmosphere model

        Args:
            altitude (float): Altitude in meters (positive up)

        Returns:
            float: Air density (kg/m^3)
        """
        if altitude < 11000:
            Temp = self.Temp_sl - self.L * altitude
        else:
            Temp = 216.65  # Stratosphere constant temperature

        rho = self.rho_sl * (Temp / self.Temp_sl)**4.256
        return rho

    def thrust_force(self, altitude, throttle):
        """
        Simple static thrust model with altitude correction

        Args:
            altitude (float): Altitude (m)
            throttle (float): Throttle setting (0-1)

        Returns:
            float: Thrust force (N)
        """
        rho = self.atmosphere(altitude)
        sigma = rho / self.rho_sl  # Density ratio
        T = self.thrust_sl * sigma * throttle
        return T

    def dynamics(self, state, controls=None):
        """
        Longitudinal aircraft dynamics equations

        Args:
            state (array): [u, w, q, theta, x, z]
            controls (dict): {'throttle': 0-1, 'delta_p': rad} 
                           If None, uses throttle=0.5, delta_p=0

        Returns:
            array: State derivatives [u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot]
        """
        # Unpack state
        u, w, q, theta, x, z = state

        # Handle controls with defaults
        if controls is None:
            throttle = 0.5
            delta_p = 0.0  # nozzle pitch angle
        else:
            throttle = controls.get('throttle', 0.5)
            delta_p = controls.get('delta_p', 0.0)

        # Current altitude (positive up, z is negative in NED)
        h = -z

        # Calculate flight conditions
        V = np.sqrt(u**2 + w**2)
        if V < 1e-6:  # Avoid division by zero
            V = 1e-6

        alpha = np.arctan2(w, u)  # Angle of attack
        rho = self.atmosphere(h)
        q_infty = 0.5 * rho * V**2  # Dynamic pressure

        # Get thrust
        T = self.thrust_force(h, throttle)

        # Aerodynamic coefficients
        CL = self.CL_alpha * alpha
        CD = self.CD_0 + self.CD_alpha * alpha**2
        CM = self.CM_alpha * alpha + self.CM_q * (q * self.chord) / (2 * V)

        # Aerodynamic forces and moments (body frame)
        Fx_aero = -q_infty * self.S * CD  # Drag (opposes motion)
        Fz_aero = -q_infty * self.S * CL  # Lift (negative in body z)
        M_aero = q_infty * self.S * self.chord * CM  # Pitch moment

        # Thrust vectoring forces and moments (body frame)
        Fx_thrust = T * np.cos(delta_p)  # Forward thrust component
        Fz_thrust = T * np.sin(delta_p)  # Vertical thrust component
        M_thrust = T * np.sin(delta_p) * self.l_arm  # Pitch moment from TVC

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
