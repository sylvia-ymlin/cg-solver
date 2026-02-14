"""Plot weak scaling from UPPMAX cluster data."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python visualize_benchmark.py <weak_scaling.tsv>")
    sys.exit(1)

tsv_file = sys.argv[1]
try:
    df = pd.read_csv(tsv_file, sep='\t')
except Exception as e:
    print(f"Error reading {tsv_file}: {e}")
    sys.exit(1)

# Sort by np to ensure correct plotting order
df = df.sort_values(by='np')
cores = df['np'].values
real_time = df['total_time_s'].values

if len(real_time) == 0:
    print("No data found in file.")
    sys.exit(1)

# Weak scaling: ideal is constant time (t1)
t1 = real_time[0]
efficiency = t1 / real_time * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(cores, real_time, 'o-', color='tab:blue', label='Measured', linewidth=2)
ax1.axhline(y=t1, color='gray', linestyle='--', label=f'Ideal ({t1:.3f}s)', alpha=0.7)
ax1.set_xlabel('Number of Processes')
ax1.set_ylabel('Execution Time (s)')
ax1.set_title('Execution Time vs. Process Count (Weak Scaling)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(cores, efficiency, 'o-', color='tab:green', label='Measured', linewidth=2)
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal')
ax2.set_xlabel('Number of Processes')
ax2.set_ylabel('Efficiency (%)')
ax2.set_title('Efficiency vs. Process Count')
ax2.set_ylim(0, 110)
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

for i, e in enumerate(efficiency):
    ax2.annotate(f"{e:.0f}%", (cores[i], e), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9)

fig.suptitle(f'Weak Scaling ({tsv_file})', fontsize=14)
fig.tight_layout()
output_file = 'docs/weak_scaling_cluster.png'
plt.savefig(output_file, dpi=300)
print(f"Saved {output_file}")
