"""
Run both standard CG and PCG to compare iterations and timing.
Usage: make && python3 scripts/run_comparative_analysis.py
"""
import subprocess
import re
import json

TOL = 1e-6
MAX_ITER = 50000
GRIDS = [64, 128, 256, 512, 1024, 2048]

results = {"cg": [], "pcg": []}

def run_benchmark(n, use_pcg):
    binary = "./PCG" if use_pcg else "./CG"
    cmd = ["mpirun", "--oversubscribe", "-np", "1", binary, str(n), str(MAX_ITER), str(TOL)]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = r.stdout
        # Match "Iterations: 100" or "Standard CG - Iterations: 100"
        it = re.search(r"Iterations:\s*(\d+)", out)
        tm = re.search(r"Time:\s*([0-9.]+)", out)
        return {"n": n, "iters": int(it.group(1)), "time": float(tm.group(1))}
    except Exception as e:
        print(f"Error at n={n}, pcg={use_pcg}: {e}")
        return None

print(f"{'n':>5} | {'CG Iters':>8} {'CG Time':>8} | {'PCG Iters':>9} {'PCG Time':>9}")
print("-" * 50)

for n in GRIDS:
    res_cg = run_benchmark(n, False)
    res_pcg = run_benchmark(n, True)
    
    if res_cg and res_pcg:
        results["cg"].append(res_cg)
        results["pcg"].append(res_pcg)
        print(f"{n:>5} | {res_cg['iters']:>8} {res_cg['time']:>8.3f} | {res_pcg['iters']:>9} {res_pcg['time']:>9.3f}")

with open("scripts/comparative_data.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results to scripts/comparative_data.json")
