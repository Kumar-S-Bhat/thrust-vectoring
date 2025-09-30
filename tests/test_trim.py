import numpy as np
from src.simulation.trim_solver import Trim_solver

trim = Trim_solver()
test_cases = [
    (150, 0, 3000),
    (250, 0, 5000),
    (200, np.radians(5), 5000),
    (200, np.radians(-3), 5000),
    (180, 0, 8000),
]

for V, gamma, h in test_cases:
    result = trim.find_trim(velocity=V, flight_path_angle=gamma, altitude=h)
    if result['success']:
        print(f"✓ V={V} γ={np.degrees(gamma):.1f}° h={h}m: "
              f"throttle={result['throttle']:.2f}, "
              f"δp={np.degrees(result['delta_p']):.1f}°, "
              f"α={np.degrees(result['alpha']):.1f}°")
    else:
        print(f"✗ V={V} γ={np.degrees(gamma):.1f}° h={h}m: FAILED")
