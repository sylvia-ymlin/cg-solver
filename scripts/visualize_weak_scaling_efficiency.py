import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def visualize_weak_scaling():
    file_path = 'results/imported/weak_scaling.tsv'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Calculate total communication time
    df['comm_ms'] = df['halo_ms'] + df['reduce_ms']
    
    # Calculate efficiency relative to np=1
    base_time = df.loc[df['np'] == 1, 'iter_time_ms'].values[0]
    df['efficiency'] = base_time / df['iter_time_ms'] * 100
    
    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute Time Breakdown
    bar_width = 0.6
    indices = np.arange(len(df['np']))
    
    ax1.bar(indices, df['comp_ms'], bar_width, label='Computation', color='#4e79a7')
    ax1.bar(indices, df['reduce_ms'], bar_width, bottom=df['comp_ms'], label='Global Reduce (Sync Overhead)', color='#e15759')
    ax1.bar(indices, df['halo_ms'], bar_width, bottom=df['comp_ms'] + df['reduce_ms'], label='Halo Exchange', color='#f28e2b')
    
    ax1.set_xlabel('Number of Processes (np)')
    ax1.set_ylabel('Time per Iteration (ms)')
    ax1.set_title('Weak Scaling: Time Breakdown')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(df['np'])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotations for Comp Time info (Memory Wall evidence)
    for i, row in df.iterrows():
        ax1.text(i, row['comp_ms'] / 2, f"{row['comp_ms']:.2f}", ha='center', color='white', fontweight='bold')
    
    # Plot 2: Relative Impact (Quantifying the Drop)
    # Normalize everything to np=1 case
    
    # Computation degradation factor
    base_comp = df.loc[df['np']==1, 'comp_ms'].values[0]
    df['comp_overhead'] = df['comp_ms'] - base_comp
    
    ax2.plot(df['np'], df['efficiency'], marker='o', linewidth=2, color='black', label='Total Weak Scaling Efficiency')
    ax2.set_xlabel('Number of Processes (np)')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Weak Scaling Efficiency')
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Create a secondary axis for overhead in ms
    ax3 = ax2.twinx()
    ax3.bar(df['np'], df['comp_overhead'], alpha=0.3, color='blue', width=2, label='Comp Time Increase (Memory Wall)')
    ax3.bar(df['np'], df['comm_ms'], alpha=0.3, color='red', width=2, bottom=df['comp_overhead'], label='Comm Overhead (Sync)')
    ax3.set_ylabel('Overhead vs np=1 (ms)')
    
    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    bars, bar_labels = ax3.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc='center right')

    plt.tight_layout()
    output_path = 'docs/weak_scaling_analysis.png'
    plt.savefig(output_path, dpi=300)
    print(f"Analysis plot saved to {output_path}")

    # Print quantitative analysis
    print("\nQuantitative Analysis:")
    print("-" * 50)
    print(f"{'np':<4} {'Total(ms)':<10} {'Comp(ms)':<10} {'Comm(ms)':<10} {'Eff(%)':<8}")
    for _, row in df.iterrows():
        print(f"{int(row['np']):<4} {row['iter_time_ms']:<10.3f} {row['comp_ms']:<10.3f} {row['comm_ms']:<10.3f} {row['efficiency']:<8.1f}")
    
    print("-" * 50)
    # Compare np=4 vs np=9
    r4 = df[df['np']==4].iloc[0]
    r9 = df[df['np']==9].iloc[0]
    
    comp_increase = r9['comp_ms'] - r4['comp_ms']
    comm_increase = r9['comm_ms'] - r4['comm_ms']
    total_increase = r9['iter_time_ms'] - r4['iter_time_ms']
    
    print(f"\nBreakpoints Analysis (np=4 -> np=9):")
    print(f"Total Time Increase: {total_increase:.3f} ms")
    print(f"  - Due to Computation (Memory Wall): {comp_increase:.3f} ms ({comp_increase/total_increase*100:.1f}%)")
    print(f"  - Due to Comm/Sync Overhead:        {comm_increase:.3f} ms ({comm_increase/total_increase*100:.1f}%)")

if __name__ == "__main__":
    visualize_weak_scaling()
