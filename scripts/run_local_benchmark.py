import subprocess
import re
import json
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark():
    grid_size = 2048 # Fixed size for Strong Scaling
    cores_list = [1, 2, 4, 8] # Run on available local cores
    results = {}
    
    print(f"Starting Local Strong Scaling Benchmark (Grid: {grid_size}x{grid_size})...")
    
    for cores in cores_list:
        print(f"Running with {cores} cores...", end="", flush=True)
        try:
            cmd = ["mpirun", "-np", str(cores), "./CG", str(grid_size)]
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = result.stdout
            
            # Parse time
            match = re.search(r"Time:\s*([0-9.]+)", output)
            if match:
                time_sec = float(match.group(1))
                results[cores] = time_sec
                print(f" Done. Time: {time_sec:.4f}s")
            else:
                print(" Failed to parse time.")
                print("Output:", output)
                
        except Exception as e:
            print(f" Error: {e}")
            
    print("Benchmark Complete.")
    print("Results:", results)
    
    # Generate Plot Code with REAL data
    generate_plot_script(results)

def generate_plot_script(results):
    cores = sorted(results.keys())
    times = [results[c] for c in cores]
    
    t1 = times[0]
    speedups = [t1/t for t in times]
    ideal_speedups = cores
    efficiency = [s/c * 100 for s, c in zip(speedups, cores)]
    
    # Overwrite the visualization script with REAL data
    content = f"""import matplotlib.pyplot as plt
import numpy as np

def plot_strong_scaling():
    # Strong Scaling: Total problem size is fixed (2048x2048).
    # Data source: Local Benchmark (Mac M-Series, {max(cores)} Cores)
    
    cores = np.array({cores})
    real_time = np.array({times})
    
    t1 = real_time[0]
    ideal_speedup = cores
    real_speedup = t1 / real_time
    efficiency = real_speedup / cores * 100

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Cores (MPI Ranks)')
    ax1.set_ylabel('Speedup (T1 / Tn)', color=color)
    ax1.plot(cores, real_speedup, 'o-', color=color, label='Measured Speedup', linewidth=2)
    ax1.plot(cores, ideal_speedup, '--', color='gray', label='Ideal (Linear Speedup)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Axes
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xticks(cores)
    ax1.set_yticks(cores)
    from matplotlib.ticker import ScalarFormatter
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(ScalarFormatter())
    
    ax1.grid(True, linestyle=':', alpha=0.6, which="both")
    ax1.legend(loc='upper left')

    # Efficiency
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Parallel Efficiency (%)', color=color)
    ax2.plot(cores, efficiency, 's--', color=color, label='Efficiency', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)
    
    for i, txt in enumerate(efficiency):
        ax2.annotate(f"{{txt:.1f}}%", (cores[i], efficiency[i]), 
                     xytext=(0, -15), textcoords='offset points', ha='center', color=color, fontweight='bold')

    plt.title('Strong Scaling Performance (Local 8-Core Benchmark)\\nFixed Total Problem Size (2048x2048)', fontsize=14)
    fig.tight_layout()
    
    plt.savefig('docs/strong_scaling.png', dpi=300)
    print("Generated docs/strong_scaling.png with REAL Local Data")

if __name__ == "__main__":
    plot_strong_scaling()
"""
    
    with open("scripts/visualize_strong_scaling.py", "w") as f:
        f.write(content)

if __name__ == "__main__":
    run_benchmark()
