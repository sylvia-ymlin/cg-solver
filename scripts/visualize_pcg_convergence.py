"""
Plot PCG convergence analysis: Total execution time vs grid size (log-log).
Verifies O(n^3) total complexity scaling for the preconditioned solver.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("scripts/pcg_convergence_data.json") as f:
    data = json.load(f)

# Use points from n=128 onwards for a cleaner slope
ns = np.array([d["n"] for d in data if d["n"] >= 128])
times = np.array([d["time"] for d in data if d["n"] >= 128])

# Linear fit in log-log space
coeffs = np.polyfit(np.log(ns), np.log(times), 1)
slope = coeffs[0]
fitted = np.exp(np.polyval(coeffs, np.log(ns)))

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ns, times, "o-", color="tab:green", markersize=8, label="PCG (Block-Jacobi) Time")
ax.loglog(ns, fitted, "--", color="tab:red", alpha=0.7,
          label=f"Fit: slope = {slope:.2f}")

# Reference O(n^3) line
ref_n = ns[-1]
ref_t = times[-1]
ref = ref_t * (ns / ref_n)**3
ax.loglog(ns, ref, ":", color="gray", alpha=0.5, label="O(n³) reference")

ax.set_xlabel("Grid size n")
ax.set_ylabel("Total Solving Time (s)")
ax.set_title("PCG Total Complexity Scaling ($O(n^3)$)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# Annotate data points
for n, t in zip(ns, times):
    ax.annotate(f"{t:.2f}s", (n, t), textcoords="offset points",
                xytext=(5, 8), fontsize=8)

plt.tight_layout()
plt.savefig("docs/pcg_convergence_scaling.png", dpi=200)
print(f"Saved docs/pcg_convergence_scaling.png")
print(f"Fitted slope: {slope:.3f} (expected ~3.0)")
