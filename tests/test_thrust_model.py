from src.propulsion.thrust_model import ThrustModel


# Simple test for verification
engine = ThrustModel()

print("--- Thrust Model Test ---")
print(f"Thrust at Sea Level (100%): {engine.thrust_force(0, 1.0):.0f} N")
print(f"Thrust at 5000m (100%):    {engine.thrust_force(5000, 1.0):.0f} N")
print(f"Thrust at 5000m (50%):     {engine.thrust_force(5000, 0.5):.0f} N")
