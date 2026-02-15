import subprocess
import os
import json
import argparse
import re
import plotting

# Configuration
RESULTS_DIR = "results/local"
BIN_CG = "./CG"
BIN_PCG = "./PCG"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_command(cmd, capture_output=True):
    """Run a shell command and return output."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True if isinstance(cmd, str) else False, 
                                  check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, shell=True if isinstance(cmd, str) else False, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(e.stderr)
        return None

def parse_time(output):
    """Parse Time from output."""
    match = re.search(r"Time: ([\d\.]+)", output)
    return float(match.group(1)) if match else None

def parse_meta(output):
    """Parse iterations, residual, error, etc."""
    meta = {}
    m_itr = re.search(r"Iterations:\s*(\d+)", output)
    m_res = re.search(r"Residual:\s*([0-9.eE+-]+)", output)
    m_err = re.search(r"L2_Error:\s*([0-9.eE+-]+)", output)
    m_time = re.search(r"Time:\s*([0-9.]+)", output)
    
    if m_itr: meta['iters'] = int(m_itr.group(1))
    if m_res: meta['residual'] = float(m_res.group(1))
    if m_err: meta['l2_error'] = float(m_err.group(1))
    if m_time: meta['time'] = float(m_time.group(1))
    return meta

def run_scaling():
    """Run Weak and Strong Scaling Benchmarks."""
    print("\n=== Running Scaling Benchmarks ===")
    
    # Files
    weak_file = os.path.join(RESULTS_DIR, "weak_scaling.tsv")
    strong_file = os.path.join(RESULTS_DIR, "strong_scaling.tsv")

    # 1. Weak Scaling
    # np -> grid_size (global)
    # 1 -> 512
    # 4 -> 1024
    # 9 -> 1536
    weak_configs = [(1, 512), (4, 1024), (9, 1536)]
    
    print("\n--- Weak Scaling ---")
    with open(weak_file, "w") as f:
        f.write("local_n\tgrid\tnp\titers\ttotal_time_s\titer_time_ms\n")
        for np_proc, grid in weak_configs:
            cmd = f"mpirun --oversubscribe -n {np_proc} {BIN_CG} {grid} 200"
            output = run_command(cmd)
            if output:
                time_s = parse_time(output)
                if time_s:
                    iter_ms = (time_s / 200) * 1000
                    f.write(f"512\t{grid}\t{np_proc}\t200\t{time_s:.6f}\t{iter_ms:.6f}\n")
                    print(f"np={np_proc}, grid={grid}, time={time_s:.4f}s")

    # 2. Strong Scaling (Fixed Global 2048x2048)
    strong_configs = [1, 4, 9]
    global_grid = 2048
    
    print("\n--- Strong Scaling ---")
    with open(strong_file, "w") as f:
        f.write("grid\tnp\titers\ttotal_time_s\titer_time_ms\n")
        for np_proc in strong_configs:
            cmd = f"mpirun --oversubscribe -n {np_proc} {BIN_CG} {global_grid} 50"
            output = run_command(cmd)
            if output:
                time_s = parse_time(output)
                if time_s:
                    iter_ms = (time_s / 50) * 1000
                    f.write(f"{global_grid}\t{np_proc}\t50\t{time_s:.6f}\t{iter_ms:.6f}\n")
                    print(f"np={np_proc}, grid={global_grid}, time={time_s:.4f}s")
    
    # Plot
    plotting.plot_weak_scaling(weak_file)
    plotting.plot_strong_scaling(strong_file)

def run_convergence():
    """Run Convergence Analysis (Residual & Total Time vs Grid Size)."""
    print("\n=== Running Convergence Analysis ===")
    
    grids = [64, 128, 256, 512, 1024, 2048]
    tol = 1e-6
    max_iter = 50000
    results = []
    
    for n in grids:
        cmd = f"mpirun -n 1 {BIN_CG} {n} {max_iter} {tol}"
        output = run_command(cmd)
        if output:
            meta = parse_meta(output)
            if meta:
                meta['n'] = n
                results.append(meta)
                print(f"n={n}, iters={meta.get('iters')}, time={meta.get('time'):.4f}s")
    
    outfile = os.path.join(RESULTS_DIR, "convergence.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    
    plotting.plot_convergence(outfile)

def run_comparison():
    """Run Comparison: CG vs PCG."""
    print("\n=== Running Solver Comparison (CG vs PCG) ===")
    
    grids = [64, 128, 256, 512, 1024, 2048]
    tol = 1e-6
    max_iter = 50000
    
    data = {"cg": [], "pcg": []}
    
    for n in grids:
        # Run CG
        cmd_cg = f"mpirun -n 1 {BIN_CG} {n} {max_iter} {tol}"
        out_cg = run_command(cmd_cg)
        meta_cg = parse_meta(out_cg) if out_cg else None
        
        # Run PCG
        cmd_pcg = f"mpirun -n 1 {BIN_PCG} {n} {max_iter} {tol}"
        out_pcg = run_command(cmd_pcg)
        meta_pcg = parse_meta(out_pcg) if out_pcg else None
        
        if meta_cg and meta_pcg:
            meta_cg['n'] = n
            meta_pcg['n'] = n
            data["cg"].append(meta_cg)
            data["pcg"].append(meta_pcg)
            print(f"n={n}: CG={meta_cg['iters']} iters, PCG={meta_pcg['iters']} iters")

    outfile = os.path.join(RESULTS_DIR, "comparison.json")
    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)
        
    plotting.plot_comparison(outfile)
    
    # Also plot ablation if historical data exists
    if os.path.exists("scripts/comparative_data.json"):
        print("Generating Pipelined CG ablation plot from historical data...")
        plotting.plot_pipelined_ablation("scripts/comparative_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CG Solver Benchmarks")
    parser.add_argument("mode", nargs="?", choices=["scaling", "convergence", "comparison", "all"], default="all",
                        help="Benchmark mode to run (default: all)")
    
    args = parser.parse_args()
    
    print("Compiling...")
    run_command("make clean && make", capture_output=False)
    
    if args.mode in ["scaling", "all"]:
        run_scaling()
    if args.mode in ["convergence", "all"]:
        run_convergence()
    if args.mode in ["comparison", "all"]:
        run_comparison()
        
    print(f"\nAll requested benchmarks complete. Results in {RESULTS_DIR}")
