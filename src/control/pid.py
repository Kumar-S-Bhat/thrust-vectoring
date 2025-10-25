import numpy as np


class PID:
    def __init__(self, kp, ki, kd, output_limits=None, derivative_filter_tau=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Output limits
        if output_limits is None:
            self.output_min = -np.inf
            self.output_max = np.inf
        else:
            self.output_min, self.output_max = output_limits

        # Derivative filter
        self.tau_d = derivative_filter_tau

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_time = None

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def update(self, setpoint, measurement, dt):
        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term with filtering
        # Raw derivative
        if dt > 0:
            raw_derivative = (error - self.prev_error) / dt
        else:
            raw_derivative = 0.0

        # First-order low-pass filter: τ*ḋ + d = d_raw
        alpha = dt / (self.tau_d + dt)
        self.filtered_derivative = (
            1 - alpha) * self.filtered_derivative + alpha * raw_derivative
        d_term = self.kd * self.filtered_derivative

        # Total output (before saturation)
        output = p_term + i_term + d_term

        # Apply output saturation
        output_saturated = np.clip(output, self.output_min, self.output_max)

        # Anti-windup: back-calculate integral to prevent windup
        if output != output_saturated:
            # We hit the limit, adjust integral
            # Simple clamping method
            self.integral = (output_saturated - p_term - d_term) / \
                self.ki if self.ki != 0 else 0.0

        # Store for next iteration
        self.prev_error = error

        return output_saturated

    def get_terms(self, setpoint, measurement):
        error = setpoint - measurement
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * self.filtered_derivative

        return {
            'error': error,
            'p': p_term,
            'i': i_term,
            'd': d_term,
            'total': p_term + i_term + d_term
        }


class CascadePID:
    def __init__(self, outer_pid, inner_scheduled_pid):
        self.outer = outer_pid
        self.inner = inner_scheduled_pid

    def reset(self):
        """Reset both controllers."""
        self.outer.reset()
        self.inner.reset()

    def update(self, outer_setpoint, outer_measurement, inner_measurement, dt):
        # Outer loop produces setpoint for inner loop
        inner_setpoint = self.outer.update(
            outer_setpoint, outer_measurement, dt)

        # Inner loop produces final control output
        control_output = self.inner.update(
            inner_setpoint, inner_measurement, outer_measurement, dt)

        return control_output
