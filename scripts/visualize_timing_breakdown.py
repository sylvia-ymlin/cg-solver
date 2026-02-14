"""
Plot timing breakdown: halo exchange vs global reduction vs computation.
Using cluster data from strong_scaling.tsv.

Usage: python3 scripts/visualize_timing_breakdown.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os

tsv_path = "results/imported/strong_scaling.tsv"
if not os.path.exists(tsv_path):
    print(f"Error: {tsv_path} not found.")
    exit(1)

# Load data from TSV
# grid	np	iters	total_time_s	iter_time_ms	halo_ms	reduce_ms	comp_ms
data = []
with open(tsv_path, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split('\t')
    for line in lines[1:]:
        if not line.strip(): continue
        vals = line.strip().split('\t')
        d = dict(zip(header, vals))
        # Convert to floats/ints
        iters = int(d['iters'])
        row = {
            'np': int(d['np']),
            'total': float(d['total_time_s']),
            't_halo': (float(d['halo_ms']) / 1000.0) * iters,
            't_reduce': (float(d['reduce_ms']) / 1000.0) * iters,
            't_comp': (float(d['comp_ms']) / 1000.0) * iters
        }
        data.append(row)

labels = [f"np={d['np']}" for d in data]
t_halo = [d["t_halo"] for d in data]
t_reduce = [d["t_reduce"] for d in data]
t_comp = [d["t_comp"] for d in data]
# Ensure other overhead is non-negative
t_other = [max(0, d["total"] - d["t_halo"] - d["t_reduce"] - d["t_comp"]) for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left: Stacked bar (absolute time)
x = np.arange(len(labels))
width = 0.5
ax1.bar(x, t_comp, width, label="Computation (stencil + dot + AXPY)", color="tab:blue")
ax1.bar(x, t_halo, width, bottom=t_comp, label="Halo Exchange (MPI_Sendrecv)", color="tab:orange")
ax1.bar(x, t_reduce, width, bottom=[c+h for c,h in zip(t_comp, t_halo)],
        label="Global Reduction (MPI_Allreduce)", color="tab:red")
ax1.bar(x, t_other, width,
        bottom=[c+h+r for c,h,r in zip(t_comp, t_halo, t_reduce)],
        label="Other overhead", color="tab:gray", alpha=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Time (s)")
ax1.set_title("Time Breakdown (n=2048, 200 iters)")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="y", alpha=0.3)

# Right: Percentage breakdown
for i, d in enumerate(data):
    total = d["total"]
    vals = [d["t_comp"]/total*100, d["t_halo"]/total*100,
            d["t_reduce"]/total*100, max(0, (total-d["t_comp"]-d["t_halo"]-d["t_reduce"]))/total*100]
    bottom = 0
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:gray"]
    names = ["Comp", "Halo", "Reduce", "Other"]
    for v, c, name in zip(vals, colors, names):
        bar = ax2.bar(i, v, width, bottom=bottom, color=c, alpha=0.8 if c != "tab:gray" else 0.4)
        if v > 3:  # label if visible
            ax2.text(i, bottom + v/2, f"{v:.1f}%", ha="center", va="center", fontsize=9)
        bottom += v

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Fraction (%)")
ax2.set_title("Time Distribution (%)")
ax2.set_ylim(0, 105)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("docs/timing_breakdown.png", dpi=200)
print("Saved docs/timing_breakdown.png")
