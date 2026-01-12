import matplotlib.pyplot as plt
import numpy as np

def plot_strong_scaling():
    # Strong Scaling: Total problem size is fixed (2048x2048).
    # Data source: Real Local Benchmark (Mac M-Series, 8 Cores)
    # Limitation: Code requires square number of processes (1, 4, 9, 16...)
    # Valid local runs: 1, 4.
    
    cores = np.array([1, 4])
    real_time = np.array([2.399, 1.248])
    
    t1 = real_time[0]
    ideal_speedup = cores
    real_speedup = t1 / real_time
    efficiency = real_speedup / cores * 100

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Cores (MPI Ranks)')
    ax1.set_ylabel('Speedup (T1 / Tn)', color=color)
    ax1.plot(cores, real_speedup, 'o-', color=color, label='Measured Speedup (Local)', linewidth=2)
    ax1.plot(cores, ideal_speedup, '--', color='gray', label='Ideal (Linear Speedup)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Axes
    ax1.set_xscale('linear') # Only 2 points, log might be weird or unnecessary
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_yticks([1, 2, 3, 4])
    from matplotlib.ticker import ScalarFormatter
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # Efficiency
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Parallel Efficiency (%)', color=color)
    ax2.plot(cores, efficiency, 's--', color=color, label='Efficiency', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)
    
    for i, txt in enumerate(efficiency):
        ax2.annotate(f"{txt:.1f}%", (cores[i], efficiency[i]), 
                     xytext=(0, -15), textcoords='offset points', ha='center', color=color, fontweight='bold')

    plt.title('Strong Scaling Performance (Local Mac Benchmark)\nFixed Total Problem Size (2048x2048)', fontsize=14)
    fig.tight_layout()
    
    plt.savefig('docs/strong_scaling.png', dpi=300)
    print("Generated docs/strong_scaling.png with REAL Local Data")

if __name__ == "__main__":
    plot_strong_scaling()
