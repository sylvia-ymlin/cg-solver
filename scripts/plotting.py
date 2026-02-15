import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd

# Standard plot settings for consistent style
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

def plot_weak_scaling(data_file='results/weak_scaling.tsv', output_file='docs/local_weak_scaling.png'):
    """Generate weak scaling plot."""
    try:
        if not os.path.exists(data_file):
            print(f"File not found: {data_file}")
            return
            
        data = np.loadtxt(data_file, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        nps = data[:, 2]
        times = data[:, 5] # iter_time_ms
        
        plt.figure(figsize=(8, 6))
        plt.plot(nps, times, 'o-', label='Local Execution')
        plt.xlabel('Number of Processes (np)')
        plt.ylabel('Time per Iteration (ms)')
        plt.title('Weak Scaling (Mac M1/Intel)')
        plt.grid(True)
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot weak scaling: {e}")

def plot_strong_scaling(data_file='results/strong_scaling.tsv', output_file='docs/local_strong_scaling.png'):
    """Generate strong scaling plot."""
    try:
        if not os.path.exists(data_file):
            print(f"File not found: {data_file}")
            return

        data = np.loadtxt(data_file, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        nps = data[:, 1]
        times = data[:, 3] # total_time_s
        
        # Speedup
        t1 = times[0]
        speedup = t1 / times
        
        plt.figure(figsize=(8, 6))
        plt.plot(nps, speedup, 'o-', color='green', label='Measured Speedup')
        plt.plot(nps, nps, '--', color='gray', label='Ideal')
        plt.xlabel('Number of Processes (np)')
        plt.ylabel('Speedup (T1/Tp)')
        plt.title('Strong Scaling (Mac M1/Intel, 2048x2048)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot strong scaling: {e}")

def plot_convergence(convergence_file='results/convergence.json', output_file='docs/convergence_analysis.png'):
    """Generate convergence analysis plots."""
    if not os.path.exists(convergence_file):
        print(f"File not found: {convergence_file}")
        return

    try:
        with open(convergence_file, 'r') as f:
            data = json.load(f)
        
        # Extract data
        ns = [row['n'] for row in data]
        total_times = [row['time'] for row in data]
        iters = [row['iters'] for row in data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Total Time vs Grid Size (Log-Log)
        ax1.loglog(ns, total_times, 'o-', linewidth=2)
        ax1.set_xlabel('Grid Size N (Total Unknowns $N^2$)')
        ax1.set_ylabel('Total Time to Solution (s)')
        ax1.set_title('Total Solving Time Scaling', fontsize=12)
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        
        # Plot 2: Iterations vs Grid Size
        ax2.plot(ns, iters, 's-', color='orange', linewidth=2)
        ax2.set_xlabel('Grid Size N')
        ax2.set_ylabel('Iterations to Convergence')
        ax2.set_title(f'Iteration Count ($O(N)$)', fontsize=12)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot convergence: {e}")

def plot_comparison(comparison_file='results/comparison.json', output_file='docs/solver_comparison.png'):
    """Generate solver comparison plots (CG vs PCG)."""
    if not os.path.exists(comparison_file):
        print(f"File not found: {comparison_file}")
        return

    try:
        with open(comparison_file, 'r') as f:
            data = json.load(f)
            
        ns = [x['n'] for x in data['cg']]
        
        # Iterations
        cg_iters = [x['iters'] for x in data['cg']]
        pcg_iters = [x['iters'] for x in data['pcg']]
        
        # Time
        cg_times = [x['time'] for x in data['cg']]
        pcg_times = [x['time'] for x in data['pcg']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Iterations (Linear scale)
        ax1.plot(ns, cg_iters, 'o-', label='Standard CG')
        ax1.plot(ns, pcg_iters, 's-', label='Preconditioned CG')
        ax1.set_xlabel('Grid Size N')
        ax1.set_ylabel('Iterations')
        ax1.set_title('Convergence: Iteration Reductions')
        ax1.legend()
        ax1.grid(True)
        
        # Subplot 2: Total Time (Log-Log scale)
        ax1.plot(ns, cg_iters, 'o-', label='Standard CG') # Redundant line? No, ax1 is done.
        
        ax2.loglog(ns, cg_times, 'o-', label='Standard CG')
        ax2.loglog(ns, pcg_times, 's-', label='Preconditioned CG')
        ax2.set_xlabel('Grid Size N (Total Unknowns $N^2$)')
        ax2.set_ylabel('Total Time (s)')
        ax2.set_title('Complexity: Total Time to Solution')
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot comparison: {e}")


def plot_pipelined_ablation(data_file='scripts/comparative_data.json', output_file='docs/pipelined_ablation.png'):
    """Generate ablation study plot for Pipelined CG."""
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        if 'pipelined' not in data:
            print("No pipelined data found.")
            return
            
        ns = [x['n'] for x in data['cg']]
        cg_times = [x['time'] for x in data['cg']]
        pipe_times = [x['time'] for x in data['pipelined']]
        
        plt.figure(figsize=(8, 6))
        plt.plot(ns, cg_times, 'o-', label='Standard CG (Baseline)')
        plt.plot(ns, pipe_times, 'x--', color='red', label='Pipelined CG (Overhead)')
        
        plt.xlabel('Grid Size N')
        plt.ylabel('Total Time (s)')
        plt.title('Ablation: Pipelined CG Overhead (Memory Bound)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot pipelined ablation: {e}")

def plot_jacobi_tuning(data_file='results/jacobi_tuning.json', output_file='docs/jacobi_tuning.png'):
    """Generate Jacobi tuning plot."""
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        steps = [x['jacobi_steps'] for x in data]
        iters = [x['iters'] for x in data]
        times = [x['time'] for x in data]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Jacobi Steps per Iteration')
        ax1.set_ylabel('Total Iterations to Convergence', color=color)
        ax1.plot(steps, iters, 'o-', color=color, label='Iterations')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:red'
        ax2.set_ylabel('Total Time (s)', color=color)  # we already handled the x-label with ax1
        ax2.plot(steps, times, 's--', color=color, label='Total Time')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Impact of Jacobi Preconditioning Steps (N=1024)')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(output_file)
        print(f"Saved {output_file}")
    except Exception as e:
        print(f"Failed to plot jacobi tuning: {e}")

if __name__ == "__main__":
    # Ensure docs directory exists
    os.makedirs('docs', exist_ok=True)
    
    print("Generating plots...")
    plot_weak_scaling()
    plot_strong_scaling()
    # Add other plot calls here as needed
