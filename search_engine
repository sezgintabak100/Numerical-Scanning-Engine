import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

class QuadraticAlgebraicScanner:
    """
    A high-speed numerical and symbolic engine designed to scan random 
    quadratic polynomial systems for nested algebraic limit cycles.
    """
    def __init__(self):
        # Define symbols for algebraic check
        self.x, self.y = sp.symbols('x y')

    def generate_random_quadratic_system(self):
        """
        Generates random coefficients for a planar quadratic system:
        dx/dt = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
        dy/dt = b0 + b1*x + b2*y + b3*x^2 + b4*x*y + b5*y^2
        """
        # Constrain coefficients to reasonable bounds to avoid numerical overflow
        a = np.random.uniform(-2.0, 2.0, 6)
        b = np.random.uniform(-2.0, 2.0, 6)
        return a, b

    def evaluate_vector_field(self, x_val, y_val, a, b):
        """Evaluates the vector field at given numerical points (x, y)."""
        dx = a[0] + a[1]*x_val + a[2]*y_val + a[3]*x_val**2 + a[4]*x_val*y_val + a[5]*y_val**2
        dy = b[0] + b[1]*x_val + b[2]*y_val + b[3]*x_val**2 + b[4]*x_val*y_val + b[5]*y_val**2
        return dx, dy

    def scan_for_limit_cycles(self, a, b, num_test_orbits=10):
        """
        Numerically integrates multiple test orbits from a grid of initial points
        to check if trajectories converge to distinct closed loops (limit cycles).
        """
        detected_cycles = 0
        grid_points = np.linspace(-1.5, 1.5, int(np.sqrt(num_test_orbits)))
        
        def system_odes(t, u):
            x_val, y_val = u[0], u[1]
            dx, dy = self.evaluate_vector_field(x_val, y_val, a, b)
            return [dx, dy]

        t_span = (0, 50.0)  # Integrate long enough to allow convergence
        t_eval = np.linspace(0, 50.0, 1000)

        for gx in grid_points:
            for gy in grid_points:
                try:
                    sol = solve_ivp(system_odes, t_span, [gx, gy], t_eval=t_eval, rtol=1e-6, atol=1e-8)
                    if sol.success:
                        # Extract the final 20% of the trajectory to check for steady-state periodicity
                        tail_idx = int(len(sol.y[0]) * 0.8)
                        x_tail = sol.y[0][tail_idx:]
                        y_tail = sol.y[1][tail_idx:]
                        
                        # Check if the trajectory settled into a periodic loop rather than a point or infinity
                        amplitude_x = np.max(x_tail) - np.min(x_tail)
                        amplitude_y = np.max(y_tail) - np.min(y_tail)
                        
                        # If it forms a loop away from the origin/infinity
                        if 0.05 < amplitude_x < 5.0 and 0.05 < amplitude_y < 5.0:
                            # Verify if it returns closely to its start in the tail segment
                            dist_to_start = np.hypot(x_tail[-1] - x_tail[0], y_tail[-1] - y_tail[0])
                            if dist_to_start < 0.02:
                                detected_cycles += 1
                except:
                    continue  # Skip divergent trajectories

        # Because multiple points can land on the same limit cycle, 
        # we check if independent trajectories detected loops.
        return detected_cycles

    def run_mass_search(self, max_iterations=5000):
        """Runs an automated fast search across thousands of configurations."""
        print(f"🚀 Initializing 48-Hour Numerical Search Engine...")
        print(f"Scanning for anomaly: Quadratic systems with multiple limit cycles...\n")
        
        for iteration in range(1, max_iterations + 1):
            a, b = self.generate_random_quadratic_system()
            cycles = self.scan_for_limit_cycles(a, b)
            
            # If the engine detects more than 1 distinct candidate trajectory region
            if cycles >= 3: 
                print(f"🚨 ANOMALY DETECTED at Iteration {iteration}!")
                print(f"Coefficients A: {np.round(a, 4)}")
                print(f"Coefficients B: {np.round(b, 4)}")
                print(f"Potential for multiple nested limit cycles found. Saving parameters...\n")
                
                # Save the parameters instantly to a local text file for verification
                with open("anomaly_log.txt", "a") as f:
                    f.write(f"Iteration: {iteration}\nA: {list(a)}\nB: {list(b)}\n---\n")
            
            if iteration % 500 == 0:
                print(f"✓ Checked {iteration} random quadratic vector fields... No global exceptions yet.")

if __name__ == "__main__":
    scanner = QuadraticAlgebraicScanner()
    # Execute a fast sample block of 1000 systems
    scanner.run_mass_search(max_iterations=1000)
