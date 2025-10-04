class ThrustModel:
    """
    Calculates engine thrust magnitude based on altitude and throttle setting,
    using a simplified ISA (International Standard Atmosphere) model.
    """

    def __init__(self, thrust_sl=130000.0, rho_sl=1.225):
        """
        Initializes atmosphere and engine constants.
        """
        # Engine constant
        self.thrust_sl = thrust_sl        # N, sea level static thrust

        # Atmospheric constants (ISA)
        self.rho_sl = rho_sl              # kg/m^3, sea level density
        self.Temp_sl = 288.15             # K, sea level temperature
        self.L = 0.0065                   # K/m, temperature lapse rate

    def atmosphere(self, altitude):
        """
        ISA atmosphere density model (up to 11km).

        Args:
            altitude (float): Altitude in meters (positive up).

        Returns:
            float: Air density (kg/m^3)
        """
        if altitude < 11000.0:
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
