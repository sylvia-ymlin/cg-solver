"""
Plot comparative convergence analysis: CG vs PCG (log-log).
Demonstrates the constant factor reduction in iteration counts.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("scripts/comparative_data.json") as f:
    data = json.load(f)

cg_ns = np.array([d["n"] for d in data["cg"] if d["n"] >= 128])
cg_iters = np.array([d["iters"] for d in data["cg"] if d["n"] >= 128])

pcg_ns = np.array([d["n"] for d in data["pcg"] if d["n"] >= 128])
pcg_iters = np.array([d["iters"] for d in data["pcg"] if d["n"] >= 128])

# Plot iterations vs n (should be O(n))
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(cg_ns, cg_iters, "o-", color="tab:blue", label="Standard CG Iterations")
ax.loglog(pcg_ns, pcg_iters, "s-", color="tab:green", label="PCG (Block-Jacobi) Iterations")

# Fit for CG
coeffs_cg = np.polyfit(np.log(cg_ns), np.log(cg_iters), 1)
ax.loglog(cg_ns, np.exp(np.polyval(coeffs_cg, np.log(cg_ns))), "--", color="tab:blue", alpha=0.5, label=f"CG slope: {coeffs_cg[0]:.2f}")

# Fit for PCG
coeffs_pcg = np.polyfit(np.log(pcg_ns), np.log(pcg_iters), 1)
ax.loglog(pcg_ns, np.exp(np.polyval(coeffs_pcg, np.log(pcg_ns))), "--", color="tab:green", alpha=0.5, label=f"PCG slope: {coeffs_pcg[0]:.2f}")

ax.set_xlabel("Grid size n")
ax.set_ylabel("Iteration Count")
ax.set_title("Numerical Scalability: CG vs PCG ($O(n)$ Iterations)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("docs/cg_vs_pcg_iterations.png", dpi=200)
print("Saved docs/cg_vs_pcg_iterations.png")

# Now plot solving time
fig2, ax2 = plt.subplots(figsize=(7, 5))

cg_times = np.array([d["time"] for d in data["cg"] if d["n"] >= 128])
pcg_times = np.array([d["time"] for d in data["pcg"] if d["n"] >= 128])

ax2.loglog(cg_ns, cg_times, "o-", color="tab:blue", label="Standard CG Time")
ax2.loglog(pcg_ns, pcg_times, "s-", color="tab:green", label="PCG (Block-Jacobi) Time")

# Fit for CG Time
coeffs_cg_t = np.polyfit(np.log(cg_ns), np.log(cg_times), 1)
ax2.loglog(cg_ns, np.exp(np.polyval(coeffs_cg_t, np.log(cg_ns))), "--", color="tab:blue", alpha=0.5, label=f"CG Time slope: {coeffs_cg_t[0]:.2f}")

# Fit for PCG Time
coeffs_pcg_t = np.polyfit(np.log(pcg_ns), np.log(pcg_times), 1)
ax2.loglog(pcg_ns, np.exp(np.polyval(coeffs_pcg_t, np.log(pcg_ns))), "--", color="tab:green", alpha=0.5, label=f"PCG Time slope: {coeffs_pcg_t[0]:.2f}")

# Reference O(n^3)
ref_n = cg_ns[-1]
ref_t = cg_times[-1]
ax2.loglog(cg_ns, ref_t * (cg_ns / ref_n)**3, ":", color="gray", alpha=0.5, label="O(n³) reference")

ax2.set_xlabel("Grid size n")
ax2.set_ylabel("Total Solving Time (s)")
ax2.set_title("Total Complexity Scaling: CG vs PCG ($O(n^3)$)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("docs/cg_vs_pcg_complexity.png", dpi=200)
print("Saved docs/cg_vs_pcg_complexity.png")
