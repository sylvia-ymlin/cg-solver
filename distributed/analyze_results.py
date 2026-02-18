#!/usr/bin/env python3
"""
Analyze distributed experiment results and generate plots.

Usage:
    python3 analyze_results.py

This script reads the TSV files from results/ and generates:
    - Strong scaling plot (speedup vs processes)
    - Weak scaling plot (time vs processes for different local sizes)
    - Node boundary analysis plot (communication ratio vs process count)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "../results"
OUTPUT_DIR = "../docs/distributed"

def read_tsv(filepath):
    """Read TSV file and return data as dict of lists."""
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Find header
    header = None
    for i, line in enumerate(lines):
        if line.startswith('np\t') or line.startswith('local_n\t'):
            header = line.strip().split('\t')
            data_lines = lines[i+1:]
            break
    
    if header is None:
        print(f"Warning: No header found in {filepath}")
        return None
    
    # Initialize columns
    for col in header:
        data[col] = []
    
    # Read data
    for line in data_lines:
        if line.strip() and not line.startswith('#'):
            values = line.strip().split('\t')
            if len(values) == len(header):
                for col, val in zip(header, values):
                    try:
                        data[col].append(float(val))
                    except ValueError:
                        data[col].append(val)
    
    return data


def plot_strong_scaling():
    """Plot strong scaling results."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "strong_scaling_*.tsv")))
    if not files:
        print("No strong scaling results found")
        return
    
    data = read_tsv(files[-1])
    if data is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    np_vals = np.array(data['np'])
    time_vals = np.array(data['time_s'])
    speedup = np.array(data['speedup'])
    efficiency = np.array(data['efficiency']) * 100
    
    # Speedup plot
    axes[0].plot(np_vals, speedup, 'bo-', linewidth=2, markersize=8, label='Measured')
    axes[0].plot(np_vals, np_vals, 'k--', linewidth=1, label='Ideal')
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Speedup')
    axes[0].set_title('Strong Scaling: Speedup')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # Efficiency plot
    axes[1].plot(np_vals, efficiency, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=100, color='k', linestyle='--', linewidth=1)
    axes[1].axhline(y=80, color='r', linestyle=':', linewidth=1, label='80% threshold')
    axes[1].set_xlabel('Number of Processes')
    axes[1].set_ylabel('Parallel Efficiency (%)')
    axes[1].set_title('Strong Scaling: Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributed_strong_scaling.png'), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/distributed_strong_scaling.png")


def plot_weak_scaling():
    """Plot weak scaling results."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "weak_scaling_*.tsv")))
    if not files:
        print("No weak scaling results found")
        return
    
    data = read_tsv(files[-1])
    if data is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by local_n
    local_sizes = sorted(set(data['local_n']))
    colors = ['b', 'g', 'r', 'm']
    
    for i, local_n in enumerate(local_sizes):
        indices = [j for j, ln in enumerate(data['local_n']) if ln == local_n]
        np_vals = [data['np'][j] for j in indices]
        time_vals = [data['time_s'][j] for j in indices]
        
        ax.plot(np_vals, time_vals, f'{colors[i]}o-', linewidth=2, markersize=8,
                label=f'n_local = {int(local_n)}')
    
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Weak Scaling: Time vs Processes\n(Constant work per process)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributed_weak_scaling.png'), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/distributed_weak_scaling.png")


def plot_node_boundary():
    """Plot node boundary analysis."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "node_boundary_*.tsv")))
    if not files:
        print("No node boundary results found")
        return
    
    data = read_tsv(files[-1])
    if data is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    np_vals = np.array(data['np'])
    halo_time = np.array(data['halo_ms'])
    allreduce_time = np.array(data['allreduce_ms'])
    stencil_time = np.array(data['stencil_ms'])
    halo_ratio = np.array(data['halo_ratio'])
    
    # Timing breakdown
    width = 0.35
    x = np.arange(len(np_vals))
    
    axes[0].bar(x - width/2, stencil_time * 1000, width, label='Stencil', color='blue')
    axes[0].bar(x + width/2, halo_time * 1000, width, label='Halo Exchange', color='red')
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Timing Breakdown')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'n={int(n)}' for n in np_vals], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Highlight node boundaries
    for i, n in enumerate(np_vals):
        if n in [20, 40, 60, 80]:
            axes[0].axvspan(i-0.5, i+0.5, alpha=0.2, color='yellow')
    
    # Halo ratio
    axes[1].plot(np_vals, halo_ratio, 'ro-', linewidth=2, markersize=8)
    axes[1].axvline(x=20, color='k', linestyle='--', label='Single node boundary')
    axes[1].axvline(x=40, color='k', linestyle='--')
    axes[1].set_xlabel('Number of Processes')
    axes[1].set_ylabel('Halo Time Ratio (%)')
    axes[1].set_title('Communication Overhead')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributed_node_boundary.png'), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/distributed_node_boundary.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Analyzing distributed experiment results...")
    print("=" * 50)
    
    plot_strong_scaling()
    plot_weak_scaling()
    plot_node_boundary()
    
    print("=" * 50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
