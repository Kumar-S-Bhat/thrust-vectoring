import numpy as np
from src.aircraft.aerodynamics import AeroTable


class ControlAllocator:
    """
    Control allocation for thrust vectoring aircraft.

    Solves: B·u = ν (Moment = Effectiveness * Control)
    where:
    - B: Control effectiveness matrix (1x2 for pitch: [B_elevator, B_nozzle])
    - u: Control vector [elevator, nozzle]
    - ν: Desired pitch moment

    Uses weighted pseudo-inverse for optimization, minimizing control effort.
    """

    def __init__(self, elevator_effectiveness=1.0, nozzle_effectiveness=1.0):
        """
        Initialize allocator.
        """
        self.B_elevator = elevator_effectiveness
        self.B_nozzle = nozzle_effectiveness

        # Control limits (radians)
        self.elevator_limits = (-np.radians(25), np.radians(25))
        self.nozzle_limits = (-np.radians(20), np.radians(20))

        self.aero = AeroTable('src/data/aero_tables.csv')

    def allocate_simple(self, desired_moment):
        """
        Simple priority allocation: Elevator first, Nozzle for residual moment.
        """

        # 1. Try elevator first
        elevator_cmd = desired_moment / self.B_elevator

        # 2. Check saturation
        if self.elevator_limits[0] <= elevator_cmd <= self.elevator_limits[1]:
            # Elevator can handle it alone
            return {
                'elevator': elevator_cmd,
                'nozzle': 0.0,
                'saturated': False,
                'residual': 0.0
            }
        else:
            # 3. Elevator saturated, use nozzle for remainder
            elevator_saturated = np.clip(elevator_cmd,
                                         self.elevator_limits[0],
                                         self.elevator_limits[1])

            moment_from_elevator = elevator_saturated * self.B_elevator
            residual_moment = desired_moment - moment_from_elevator

            # 4. Calculate nozzle command
            nozzle_cmd = residual_moment / self.B_nozzle
            nozzle_cmd = np.clip(nozzle_cmd,
                                 self.nozzle_limits[0],
                                 self.nozzle_limits[1])

            moment_from_nozzle = nozzle_cmd * self.B_nozzle
            final_residual = desired_moment - moment_from_elevator - moment_from_nozzle

            return {
                'elevator': elevator_saturated,
                'nozzle': nozzle_cmd,
                'saturated': True,
                'residual': final_residual
            }

    def allocate_weighted(self, desired_moment, weights=None):
        """
        Weighted pseudo-inverse allocation (Optimal method).

        Solution: u = W⁻¹·Bᵀ·(B·W⁻¹·Bᵀ)⁻¹·ν
        """
        if weights is None:
            weights = np.array([1.0, 2.0])  # Default: penalize nozzle more

        # 1. Setup Matrices
        B = np.array([[self.B_elevator, self.B_nozzle]])
        W_inv = np.diag(1.0 / weights)

        # 2. Calculate Weighted Pseudo-Inverse Solution
        temp = B @ W_inv @ B.T
        if temp[0, 0] < 1e-12:
            # Fallback for numerical singularity
            return self.allocate_simple(desired_moment)

        # u = W⁻¹·Bᵀ·(B·W⁻¹·Bᵀ)⁻¹·ν
        u = W_inv @ B.T @ np.linalg.inv(temp) @ np.array([desired_moment])

        # Note: u is a 2x1 array, extract scalars
        elevator_cmd, nozzle_cmd = u

        # 3. Apply limits (Saturation)
        elevator_saturated = np.clip(elevator_cmd,
                                     self.elevator_limits[0],
                                     self.elevator_limits[1])
        nozzle_saturated = np.clip(nozzle_cmd,
                                   self.nozzle_limits[0],
                                   self.nozzle_limits[1])

        # 4. Check for saturation
        saturated = (abs(elevator_cmd - elevator_saturated) > 1e-6 or
                     abs(nozzle_cmd - nozzle_saturated) > 1e-6)

        # 5. Calculate residual error
        moment_achieved = (elevator_saturated * self.B_elevator +
                           nozzle_saturated * self.B_nozzle)
        residual = desired_moment - moment_achieved

        return {
            'elevator': elevator_saturated,
            'nozzle': nozzle_saturated,
            'saturated': saturated,
            'residual': residual
        }

    def update_effectiveness(self, dynamic_pressure, alpha, thrust):
        """
        Update B_elevator and B_nozzle based on current flight conditions.
        """
        S = 27.87
        c = 3.45
        l_arm = 6.0
        CL, CD, CM_static, Cm_de = self.aero.get_coefficients(alpha)

        self.B_elevator = dynamic_pressure * S * c * Cm_de

        # B_nozzle: Depends only on thrust
        self.B_nozzle = thrust * l_arm
