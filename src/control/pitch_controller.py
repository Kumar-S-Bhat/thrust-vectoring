import numpy as np
from src.control.pid import PID, CascadePID
from src.control.allocator import ControlAllocator
from src.control.gain_scheduled import GainScheduledPID


class LongitudinalController:
    """
    Implements the full longitudinal control system:
    - Outer Loop: Angle of Attack (α) tracking (Alpha PID -> Pitch Rate Command)
    - Inner Loop: Pitch Rate (q) tracking (Rate PID -> Moment Command)
    - Allocation: Moment Command -> [Elevator, Nozzle]
    """

    def __init__(self, kp_a=3.0, ki_a=20.0, kd_a=0.056):
        """
        Initializes PID controllers and the Control Allocator.
        """

        # 1. Outer Loop PID (Alpha to Rate Command)
        self.alpha_pid = PID(
            kp=kp_a,
            ki=ki_a,
            kd=kd_a,
            # Output limits the commanded pitch rate (rad/s)
            output_limits=(-np.radians(80), np.radians(80))
        )

        # 2. Inner Loop PID (Rate to Moment Command)
        self.q_pid = GainScheduledPID(output_limits=(-100000000, 100000000))

        # 3. Cascade Structure (Alpha PID -> Rate PID)
        self.cascade = CascadePID(self.alpha_pid, self.q_pid)

        # 4. Control Allocator
        self.allocator = ControlAllocator(elevator_effectiveness=1.0,
                                          nozzle_effectiveness=1.0)

    def reset(self):
        """Reset all controller integrators."""
        self.cascade.reset()

    def update(self, dynamic_pressure, thrust, dt, alpha_meas, q_meas, alpha_cmd=None, q_cmd=None, weights=None):
        """
        Updates the control system based on feedback.

        Args:
            dynamic_pressure (float): Current q̄
            thrust (float): Current thrust (N)
            dt (float): Time step (s)
            alpha_meas (float): Actual Angle of Attack (rad)
            q_meas (float): Actual Pitch Rate (rad/s)
            alpha_cmd (float, optional): Commanded Angle of Attack (for Cobra)
            q_cmd (float, optional): Commanded Pitch Rate (for testing/tuning)

        Returns:
            dict: Final commanded control surfaces and diagnostics.
        """

        # 1. Update allocator effectiveness (MUST happen before PID runs)
        self.allocator.update_effectiveness(
            dynamic_pressure, alpha_meas, thrust)

        # 2. Generate Moment Command
        desired_moment = 0.0
        q_target = 0.0

        if alpha_cmd is not None:
            # Full Cascade Mode (Alpha -> Rate -> Moment)
            desired_moment = self.cascade.update(
                alpha_cmd, alpha_meas, q_meas, dt)

            # Extract q_target from the output of the alpha PID for logging
            q_target = self.alpha_pid.get_terms(alpha_cmd, alpha_meas)['total']

        elif q_cmd is not None:
            # Rate-only Mode (Rate -> Moment)
            desired_moment = -self.q_pid.update(q_cmd, q_meas, alpha_meas, dt)
            q_target = q_cmd

        # 3. Allocation (Moment -> Elevator/Nozzle)
        allocation = self.allocator.allocate_weighted(desired_moment, weights)

        return {
            'moment_cmd': desired_moment,
            'q_target': q_target,  # Commanded rate (output of alpha loop)
            'elevator_cmd': allocation['elevator'],
            'nozzle_cmd': allocation['nozzle'],
            'saturated': allocation['saturated'],
            'residual': allocation['residual'],
        }
