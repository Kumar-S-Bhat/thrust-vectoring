import numpy as np


def C_nb(phi, theta, psi):
    """
    Direction cosine matrix from NED to Body frame.

    Args:
        phi (float): Roll angle (rad)
        theta (float): Pitch angle (rad) 
        psi (float): Yaw angle (rad)

    Returns:
        np.array: 3x3 rotation matrix C_nb
    """
    # Individual rotation matrices
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    # Roll matrix (rotation about x-axis)
    R_x = np.array([[1, 0, 0],
                    [0, c_phi, s_phi],
                    [0, -s_phi, c_phi]])

    # Pitch matrix (rotation about y-axis)
    R_y = np.array([[c_theta, 0, -s_theta],
                    [0, 1, 0],
                    [s_theta, 0, c_theta]])

    # Yaw matrix (rotation about z-axis)
    R_z = np.array([[c_psi, s_psi, 0],
                    [-s_psi, c_psi, 0],
                    [0, 0, 1]])

    # Combined rotation: NED to Body = R_x * R_y * R_z
    C_nb = R_x @ R_y @ R_z

    return C_nb


def euler_rates(p, q, r, phi, theta):
    """
    Convert body angular rates to Euler angle rates.

    Args:
        p (float): Roll rate (rad/s)
        q (float): Pitch rate (rad/s)
        r (float): Yaw rate (rad/s)
        phi (float): Roll angle (rad)
        theta (float): Pitch angle (rad)

    Returns:
        tuple: (phi_dot, theta_dot, psi_dot) in rad/s
    """
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    t_theta = np.tan(theta)

    # Avoid singularity at theta = ±90°
    if abs(c_theta) < 1e-6:
        raise ValueError("Gimbal lock: theta too close to ±90 degrees")

    # Transformation matrix
    phi_dot = p + q * s_phi * t_theta + r * c_phi * t_theta
    theta_dot = q * c_phi - r * s_phi
    psi_dot = (q * s_phi + r * c_phi) / c_theta

    return phi_dot, theta_dot, psi_dot


def body_to_ned_velocity(u, v, w, phi, theta, psi):
    """
    Transform body-frame velocities to NED frame.

    Args:
        u, v, w (float): Body frame velocities (m/s)
        phi, theta, psi (float): Euler angles (rad)

    Returns:
        tuple: (x_dot, y_dot, z_dot) NED velocities (m/s)
    """
    C = C_nb(phi, theta, psi)
    vel_body = np.array([u, v, w])
    vel_ned = C.T @ vel_body  # C.T transforms Body to NED

    return vel_ned[0], vel_ned[1], vel_ned[2]


def angle_of_attack_sideslip(u, v, w):
    """
    Calculate angle of attack and sideslip from body velocities.

    Args:
        u, v, w (float): Body frame velocities (m/s)

    Returns:
        tuple: (alpha, beta, V_total) where angles in rad, V in m/s
    """
    V_total = np.sqrt(u**2 + v**2 + w**2)

    if V_total < 1e-6:
        return 0.0, 0.0, 0.0

    alpha = np.arctan2(w, u)  # Angle of attack
    beta = np.arcsin(v / V_total)  # Sideslip angle

    return alpha, beta, V_total
