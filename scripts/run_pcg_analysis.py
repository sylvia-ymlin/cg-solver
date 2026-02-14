"""
Run CG to convergence for various grid sizes.
Reports iteration count, L2 error, and timing breakdown.

Usage: make && python3 scripts/run_convergence_analysis.py
"""
import subprocess
import re
import json

TOL = 1e-6
MAX_ITER = 50000
GRIDS = [64, 128, 256, 512, 1024, 2048]

results = []

print(f"Convergence analysis (tol={TOL})\n")
print(f"{'n':>5}  {'iters':>6}  {'residual':>14}  {'L2_error':>14}  {'time':>8}")
print("-" * 60)

for n in GRIDS:
    cmd = ["mpirun", "--oversubscribe", "-np", "1", "./CG",
           str(n), str(MAX_ITER), str(TOL)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = r.stdout
        it = re.search(r"Iterations:\s*(\d+)", out)
        res = re.search(r"Residual:\s*([0-9.eE+-]+)", out)
        err = re.search(r"L2_Error:\s*([0-9.eE+-]+)", out)
        tm = re.search(r"Time:\s*([0-9.]+)", out)
        th = re.search(r"Time_Halo:\s*([0-9.]+)", out)
        tr = re.search(r"Time_Reduce:\s*([0-9.]+)", out)
        tc = re.search(r"Time_Comp:\s*([0-9.]+)", out)

        row = {
            "n": n, "h": 1.0/(n+1),
            "iters": int(it.group(1)),
            "residual": float(res.group(1)),
            "l2_error": float(err.group(1)),
            "time": float(tm.group(1)),
            "t_halo": float(th.group(1)),
            "t_reduce": float(tr.group(1)),
            "t_comp": float(tc.group(1)),
        }
        results.append(row)
        print(f"{n:>5}  {row['iters']:>6}  {row['residual']:>14.6e}  {row['l2_error']:>14.6e}  {row['time']:>8.3f}")
    except subprocess.TimeoutExpired:
        print(f"{n:>5}  TIMEOUT (>{600}s)")
    except Exception as e:
        print(f"{n:>5}  ERROR: {e}")

# Save for plotting
with open("scripts/pcg_convergence_data.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} data points to scripts/pcg_convergence_data.json")
