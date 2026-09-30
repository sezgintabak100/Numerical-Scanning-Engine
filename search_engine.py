import os
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class HighPerformanceAlgebraicScanner:
    """
    An advanced numerical-symbolic engine that automatically scans quadratic fields,
    solves for invariant algebraic curves using SymPy, and saves visual plots of anomalies.
    """
    def __init__(self):
        # Define symbolic variables for algebraic matching
        self.x, self.y = sp.symbols('x y', real=True)
        
        # Define the template for a generic degree-2 invariant curve: f(x,y) = 0
        self.c = sp.symbols('c0:6', real=True) # Curve coefficients c0 to c5
        self.f_curve = (self.c[0] + self.c[1]*self.x + self.c[2]*self.y + 
                        self.c[3]*self.x**2 + self.c[4]*self.x*self.y + self.c[5]*self.y**2)

    def generate_random_quadratic_system(self):
        """Generates bounded coefficients for a random planar quadratic system."""
        a = np.random.uniform(-1.5, 1.5, 6)
        b = np.random.uniform(-1.5, 1.5, 6)
        return a, b

    def evaluate_vector_field(self, x_val, y_val, a, b):
        """Evaluates the vector field components numerically."""
        dx = a[0] + a[1]*x_val + a[2]*y_val + a[3]*x_val**2 + a[4]*x_val*y_val + a[5]*y_val**2
        dy = b[0] + b[1]*x_val + b[2]*y_val + b[3]*x_val**2 + b[4]*x_val*y_val + b[5]*y_val**2
        return dx, dy

    def solve_algebraic_curve(self, a, b):
        """
        Uses SymPy to solve the Darboux invariance condition: P*(df/dx) + Q*(df/dy) = K*f
        where K(x,y) = k0 + k1*x + k2*y is a linear polynomial cofactor for quadratic systems.
        """
        # 1. Define vector field symbolically
        P = a[0] + a[1]*self.x + a[2]*self.y + a[3]*self.x**2 + a[4]*self.x*self.y + a[5]*self.y**2
        Q = b[0] + b[1]*self.x + b[2]*self.y + b[3]*self.x**2 + b[4]*self.x*self.y + b[5]*self.y**2
        
        # 2. Compute partial derivatives of the curve
        df_dx = sp.diff(self.f_curve, self.x)
        df_dy = sp.diff(self.f_curve, self.y)
        
        # 3. Setup linear cofactor template (degree-1 polynomial since field is degree-2 and curve is degree-2)
        k0, k1, k2 = sp.symbols('k0 k1 k2', real=True)
        K = k0 + k1*self.x + k2*self.y
        
        # 4. Formulate the Darboux identity equation
        identity = sp.expand(P * df_dx + Q * df_dy - K * self.f_curve)
        
        # 5. Extract polynomial coefficients relative to x and y monomials to force the identity to 0
        poly_eqs = sp.Poly(identity, self.x, self.y).coeffs()
        
        # 6. Solve the resulting system of algebraic equations for the curve parameters (c0..c5) and cofactor (k0..k2)
        all_variables = list(self.c) + [k0, k1, k2]
        solutions = sp.solve(poly_eqs, all_variables, dict=True)
        
        valid_curves = []
        for sol in solutions:
            # Substitute solution back into our general curve equation
            specific_curve = self.f_curve.subs(sol)
            # Filter out the trivial solution (where the curve evaluates identically to zero)
            if specific_curve != 0 and any(c_val in sol for c_val in self.c):
                valid_curves.append(specific_curve)
                
        return valid_curves

    def save_anomaly_plot(self, iteration, a, b, trajectories):
        """Extension: Plots the flow landscape and saves a clean image file instantly."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Generate background streamline flow mesh
        Y, X = np.mgrid[-2:2:20j, -2:2:20j]
        U, V = self.evaluate_vector_field(X, Y, a, b)
        
        ax.streamplot(X, Y, U, V, color='cornflowerblue', linewidth=0.8, density=1.2, arrowstyle='->')
        
        # Overlay the tracked periodic trajectories that flagged the anomaly
        for traj in trajectories:
            ax.plot(traj[0], traj[1], color='crimson', linewidth=2, label='Periodic Orbit')
            
        ax.set_title(f"Anomaly Phase Portrait - System #{iteration}", fontsize=12, fontweight='bold')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Make a directory for outputs if it doesn't exist
        os.makedirs("anomalies", exist_ok=True)
        file_path = f"anomalies/system_{iteration}.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📸 Phase portrait visual successfully generated and saved to: {file_path}")

    def scan_for_limit_cycles(self, a, b, num_test_orbits=16):
        """Numerically integrates a grid of initial paths to check for persistent loops."""
        detected_cycles = 0
        grid_vals = np.linspace(-1.2, 1.2, int(np.sqrt(num_test_orbits)))
        valid_trajectories = []
        
        def system_odes(t, u):
            dx, dy = self.evaluate_vector_field(u[0], u[1], a, b)
            return [dx, dy]

        t_span = (0, 40.0)
        t_eval = np.linspace(0, 40.0, 600)

        for gx in grid_vals:
            for gy in grid_vals:
                try:
                    sol = solve_ivp(system_odes, t_span, [gx, gy], t_eval=t_eval, rtol=1e-5, atol=1e-7)
                    if sol.success:
                        tail_idx = int(len(sol.y[0]) * 0.8)
                        x_tail = sol.y[0][tail_idx:]
                        y_tail = sol.y[1][tail_idx:]
                        
                        amp_x = np.max(x_tail) - np.min(x_tail)
                        amp_y = np.max(y_tail) - np.min(y_tail)
                        
                        # Isolate genuine limit cycles from stationary points or explosive infinity bounds
                        if 0.1 < amp_x < 4.0 and 0.1 < amp_y < 4.0:
                            dist = np.hypot(x_tail[-1] - x_tail, y_tail[-1] - y_tail)
                            if dist[0] < 0.05:
                                detected_cycles += 1
                                valid_trajectories.append((sol.y[0], sol.y[1]))
                except:
                    continue
        return detected_cycles, valid_trajectories

    def execute_search(self, max_runs=1000):
        """Runs the complete automation loop combining numeric limits and symbolic curves."""
        print("🚀 Initializing Enhanced Symbolic-Numeric Search Engine...")
        print("Scanning for quadratic vector fields with distinct limit cycles & invariant curves...\n")
        
        for iteration in range(1, max_runs + 1):
            a, b = self.generate_random_quadratic_system()
            cycles, trajectories = self.scan_for_limit_cycles(a, b)
            
            # Anomaly trigger: The system shows active trapping regions/periodic orbits
            if cycles >= 2:
                print(f"\n🚨 POTENTIAL ANOMALY DETECTED AT ITERATION {iteration}!")
                print(f"Coefficients A (dx/dt): {np.round(a, 3)}")
                print(f"Coefficients B (dy/dt): {np.round(b, 3)}")
                
                # Instantly execute the algebraic solver to find invariant degree-2 curves
                print("🧠 Running symbolic Darboux solver via SymPy...")
                found_curves = self.solve_algebraic_curve(a, b)
                
                if found_curves:
                    print(f"✅ Found {len(found_curves)} Invariant Quadratic Algebraic Curve(s):")
                    for idx, curve in enumerate(found_curves):
                        print(f"   Curve {idx+1}: {curve} = 0")
                else:
                    print("❌ No explicit invariant degree-2 algebraic curves found for this configuration.")
                
                # Execute the plotting extension
                self.save_anomaly_plot(iteration, a, b, trajectories)
                
                # Log data to local storage
                with open("anomaly_findings.txt", "a") as f:
                    f.write(f"System #{iteration}\nA: {list(a)}\nB: {list(b)}\nCurves: {found_curves}\n---\n")
            
            if iteration % 100 == 0:
                print(f"✓ Swept through {iteration} random fields...")

if __name__ == "__main__":
    scanner = HighPerformanceAlgebraicScanner()
    # Run a test execution loop of 500 configurations
    scanner.execute_search(max_runs=500)
