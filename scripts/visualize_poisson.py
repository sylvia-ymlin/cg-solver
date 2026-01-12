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
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    X, Y, U = generate_exact_solution(100)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, U, cmap='viridis',
                          linewidth=0, antialiased=False, alpha=0.9)
    
    # Customize axis
    ax.set_zlim(0, np.max(U)*1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.view_init(elev=30, azim=225)  # Preferred angle
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5, label='u(x,y)')
    
    plt.title('Expected Solution: Poisson 2D\n$u(x,y) = x(1-x)y(1-y)$', fontsize=12)
    
    # Save
    plt.savefig('docs/solution_viz.png', dpi=300, bbox_inches='tight')
    print("Generated docs/solution_viz.png")

if __name__ == "__main__":
    import os
    if not os.path.exists('docs'):
        os.makedirs('docs')
    plot_solution()
