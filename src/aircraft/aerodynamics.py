import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os


class AeroTable:
    """
    Aerodynamic coefficient lookup table class using CubicSpline.
    """

    def __init__(self, data_file=None):
        self.data_file = data_file
        self.CL_interp = None
        self.CD_interp = None
        self.Cm_interp = None
        self.Cm_de_interp = None

        self.alpha_min_rad = 0.0
        self.alpha_max_rad = 0.0

        if data_file and os.path.exists(data_file):
            self.load_data(data_file)

    def load_data(self, data_file):
        """
        Loads CSV, sorts by alpha, and calls the interpolation method.
        """
        df = pd.read_csv(data_file)
        df_sorted = df.sort_values(by='alpha_degree').reset_index(drop=True)
        self.data_interpolate(df_sorted)

    def data_interpolate(self, df_sorted):

        alpha_radian = np.radians(df_sorted['alpha_degree'].values)

        # Store the boundary values
        self.alpha_min_rad = alpha_radian[0]
        self.alpha_max_rad = alpha_radian[-1]

        for coeff_name in ['CL', 'CD', 'Cm', 'Cm_de']:
            interp = CubicSpline(alpha_radian, df_sorted[coeff_name].values)
            setattr(self, f'{coeff_name}_interp', interp)

    def get_coefficients(self, alpha, beta=0.0):
        """
        Get aerodynamic coefficients via interpolation.

        Args:
            alpha (float): Angle of attack (RADIAN)
            beta (float): Sideslip angle (RADIAN) - Currently ignored (1D)

        Returns:
            list: [CL, CD, Cm, Cm_de]
        """

        if self.CL_interp is None:
            raise ValueError('Table is not provided')

        # CLAMPING MECHANISM (Prevents Wild Extrapolation)
        # forces 'alpha' to stay within the safe range
        alpha = np.clip(
            alpha,
            self.alpha_min_rad,
            self.alpha_max_rad
        )

        CL = float(self.CL_interp(alpha))
        CD = float(self.CD_interp(alpha))
        Cm = float(self.Cm_interp(alpha))
        Cm_de = float(self.Cm_de_interp(alpha))

        return [CL, CD, Cm, Cm_de]
