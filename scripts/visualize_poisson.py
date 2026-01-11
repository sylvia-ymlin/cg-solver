import numpy as np
import matplotlib.pyplot as plt

def generate_exact_solution(n=100):
    """
    Generates the exact solution for the Poisson equation:
    -Delta u = f
    with u = x(1-x)y(1-y) on unit square.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Exact solution
    U = X * (1 - X) * Y * (1 - Y)
    
    return X, Y, U

def plot_solution():
    X, Y, U = generate_exact_solution(200)
    
    plt.figure(figsize=(10, 8))
    # Use a nice colormap (jet or viridis)
    cp = plt.contourf(X, Y, U, 50, cmap='viridis')
    plt.colorbar(cp, label='u(x,y)')
    plt.title('Expected Solution: Poisson 2D\n$u(x,y) = x(1-x)y(1-y)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Save
    plt.savefig('docs/solution_viz.png', dpi=300, bbox_inches='tight')
    print("Generated docs/solution_viz.png")

if __name__ == "__main__":
    import os
    if not os.path.exists('docs'):
        os.makedirs('docs')
    plot_solution()
