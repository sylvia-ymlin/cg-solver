import subprocess
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = "results/local"
BIN_NAME = "./CG"
WEAK_SCALING_FILE = os.path.join(RESULTS_DIR, "weak_scaling.tsv")
STRONG_SCALING_FILE = os.path.join(RESULTS_DIR, "strong_scaling.tsv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_command(cmd):
    """Run a shell command and return output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(e.stderr)
        return None

def parse_output(output):
    """Parse the standard output of the CG solver."""
    # Expected: "Standard CG - Iterations: 200, Time: 0.123456"
    import re
    match = re.search(r"Time: ([\d\.]+)", output)
    if match:
        return float(match.group(1))
    return None

def benchmark():
    # 1. Compile
    print("Compiling...")
    run_command("make clean && make")

    # 2. Weak Scaling (local 512x512)
    # np -> grid_size (global)
    # 1 -> 512
    # 4 -> 1024
    # 9 -> 1536
    # 16 -> 2048 (Oversubscription on 8-core Mac, good for 'Context Switch' arg)
    weak_configs = [
        (1, 512),
        (4, 1024),
        (9, 1536)
    ]
    
    print("\n--- Running Weak Scaling ---")
    with open(WEAK_SCALING_FILE, "w") as f:
        f.write("local_n\tgrid\tnp\titers\ttotal_time_s\titer_time_ms\n")
        
        for np_proc, grid in weak_configs:
            # Use --oversubscribe to allow running more processes than cores (crucial for np=9 on 8-core Mac)
            cmd = f"mpirun --oversubscribe -n {np_proc} {BIN_NAME} {grid} 200"
            output = run_command(cmd)
            if output:
                time_s = parse_output(output)
                if time_s:
                    iter_ms = (time_s / 200) * 1000
                    f.write(f"512\t{grid}\t{np_proc}\t200\t{time_s:.6f}\t{iter_ms:.6f}\n")
                    print(f"np={np_proc}, grid={grid}, time={time_s:.4f}s")

    # 3. Strong Scaling (Fixed Global 2048x2048)
    # np: 1, 4, 9
    strong_configs = [1, 4, 9]
    global_grid = 2048
    
    print("\n--- Running Strong Scaling ---")
    with open(STRONG_SCALING_FILE, "w") as f:
        f.write("grid\tnp\titers\ttotal_time_s\titer_time_ms\n")
        
        for np_proc in strong_configs:
            cmd = f"mpirun --oversubscribe -n {np_proc} {BIN_NAME} {global_grid} 50" # 50 iters for speed
            output = run_command(cmd)
            if output:
                time_s = parse_output(output)
                if time_s:
                    iter_ms = (time_s / 50) * 1000
                    f.write(f"{global_grid}\t{np_proc}\t50\t{time_s:.6f}\t{iter_ms:.6f}\n")
                    print(f"np={np_proc}, grid={global_grid}, time={time_s:.4f}s")
                    
    print("\nBenchmark complete. Results saved to results/local/")

def plot_results():
    # Simple Plotting logic for newly generated data
    
    # Weak Scaling
    try:
        data = np.loadtxt(WEAK_SCALING_FILE, skiprows=1)
        # Handle case with only 1 row (1D array)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        nps = data[:, 2]
        times = data[:, 5] # iter_time_ms
        
        plt.figure(figsize=(8, 6))
        plt.plot(nps, times, 'o-', linewidth=2, label='Local Execution')
        plt.xlabel('Number of Processes (np)')
        plt.ylabel('Time per Iteration (ms)')
        plt.title('Weak Scaling (Mac M1/Intel)')
        plt.grid(True)
        plt.savefig('docs/local_weak_scaling.png')
        print("Saved docs/local_weak_scaling.png")
    except Exception as e:
        print(f"Failed to plot weak scaling: {e}")

    # Strong Scaling
    try:
        data = np.loadtxt(STRONG_SCALING_FILE, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        nps = data[:, 1]
        times = data[:, 3] # total_time_s
        
        # Speedup
        t1 = times[0]
        speedup = t1 / times
        
        plt.figure(figsize=(8, 6))
        plt.plot(nps, speedup, 'o-', linewidth=2, color='green', label='Measured Speedup')
        plt.plot(nps, nps, '--', color='gray', label='Ideal')
        plt.xlabel('Number of Processes (np)')
        plt.ylabel('Speedup (T1/Tp)')
        plt.title('Strong Scaling (Mac M1/Intel, 2048x2048)')
        plt.legend()
        plt.grid(True)
        plt.savefig('docs/local_strong_scaling.png')
        print("Saved docs/local_strong_scaling.png")
    except Exception as e:
        print(f"Failed to plot strong scaling: {e}")

if __name__ == "__main__":
    benchmark()
    plot_results()
