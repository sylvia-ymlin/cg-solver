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

results = {"cg": [], "pcg": [], "pipelined": []}

def run_benchmark(n, mode):
    # mode: 0=CG, 1=PCG, 2=Pipelined
    if mode == 0: binary = "./CG"
    elif mode == 1: binary = "./PCG"
    else: binary = "./PipelinedCG"
    
    cmd = ["mpirun.mpich", "-np", "1", binary, str(n), str(MAX_ITER), str(TOL)]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = r.stdout
        # Match "Iterations: 100" or "Standard CG - Iterations: 100"
        it = re.search(r"Iterations:\s*(\d+)", out)
        tm = re.search(r"Time:\s*([0-9.]+)", out)
        return {"n": n, "iters": int(it.group(1)), "time": float(tm.group(1))}
    except Exception as e:
        print(f"Error at n={n}, mode={mode}: {e}")
        return None

print(f"{'n':>5} | {'CG':>12} | {'PCG':>12} | {'PipeCG':>12}")
print("-" * 60)

for n in GRIDS:
    res_cg = run_benchmark(n, 0)
    res_pcg = run_benchmark(n, 1)
    res_pipe = run_benchmark(n, 2)
    
    if res_cg and res_pcg and res_pipe:
        results["cg"].append(res_cg)
        results["pcg"].append(res_pcg)
        results["pipelined"].append(res_pipe)
        print(f"{n:>5} | {res_cg['iters']:>4}/{res_cg['time']:>6.3f} | {res_pcg['iters']:>4}/{res_pcg['time']:>6.3f} | {res_pipe['iters']:>4}/{res_pipe['time']:>6.3f}")

with open("scripts/comparative_data.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results to scripts/comparative_data.json")
