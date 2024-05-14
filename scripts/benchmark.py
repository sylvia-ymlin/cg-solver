import subprocess
import os
import json
import argparse
import re

import plotting

RESULTS_DIR = "results"
BIN = "./solver"

CONVERGENCE_NS = [64, 128, 256, 512, 1024, 2048]
JACOBI_STEPS = [1, 5, 10, 20]
JACOBI_GRID_N = 1024

os.makedirs(RESULTS_DIR, exist_ok=True)


def mpi_cmd(np, args):
    return f"mpirun --oversubscribe -n {np} {BIN} {args}"


def run(cmd):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None


def parse_meta(output):
    meta = {}
    m = re.search(r"Iterations:\s*(\d+)", output)
    if m:
        meta['iters'] = int(m.group(1))
    m = re.search(r"Total Time:\s*([0-9.]+)", output)
    if m:
        meta['time'] = float(m.group(1))
    return meta


def run_scaling():
    print("\n=== Scaling ===")

    weak_results = []
    # Primary points are <= 25; 36 is kept as an oversubscription comparison point.
    weak_points = [(1, 512), (4, 1024), (9, 1536), (16, 2048), (25, 2560), (36, 3072)]
    for np, grid in weak_points:
        out = run(mpi_cmd(np, f"{grid} 200 0.0 0"))
        if out:
            meta = parse_meta(out)
            meta.update({'np': np, 'grid': grid, 'local_n': 512, 'is_supplemental': np == 36})
            weak_results.append(meta)
            label = " [supplemental]" if np == 36 else ""
            print(f"weak: np={np}, grid={grid}, time={meta['time']:.4f}s{label}")

    strong_results = []
    # Keep 36 as supplemental to show behavior beyond available physical cores.
    strong_points = [1, 4, 9, 16, 25, 36]
    for np in strong_points:
        out = run(mpi_cmd(np, "2048 50 0.0 0"))
        if out:
            meta = parse_meta(out)
            meta.update({'np': np, 'grid': 2048, 'is_supplemental': np == 36})
            strong_results.append(meta)
            label = " [supplemental]" if np == 36 else ""
            print(f"strong: np={np}, time={meta['time']:.4f}s{label}")

    with open(f"{RESULTS_DIR}/weak_scaling.json", "w") as f:
        json.dump(weak_results, f, indent=2)
    with open(f"{RESULTS_DIR}/strong_scaling.json", "w") as f:
        json.dump(strong_results, f, indent=2)

    plotting.plot_weak_scaling()
    plotting.plot_strong_scaling()


def run_convergence():
    print("\n=== Convergence ===")
    results = []
    for n in CONVERGENCE_NS:
        out = run(mpi_cmd(1, f"{n} 50000 1e-6 0"))
        if out:
            meta = parse_meta(out)
            meta['n'] = n
            results.append(meta)
            print(f"n={n}, iters={meta['iters']}, time={meta['time']:.4f}s")

    outfile = f"{RESULTS_DIR}/convergence.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    plotting.plot_convergence(outfile)


def run_jacobi_tuning():
    print("\n=== Jacobi Tuning ===")
    results = []
    for steps in JACOBI_STEPS:
        out = run(mpi_cmd(1, f"{JACOBI_GRID_N} 50000 1e-6 {steps}"))
        if out:
            meta = parse_meta(out)
            meta['jacobi_steps'] = steps
            results.append(meta)
            print(f"steps={steps}, iters={meta['iters']}, time={meta['time']:.4f}s")

    outfile = f"{RESULTS_DIR}/jacobi_tuning.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    plotting.plot_jacobi_tuning(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?",
                        choices=["scaling", "convergence", "jacobi_tuning", "all"],
                        default="all")
    args = parser.parse_args()

    run("make clean && make")

    if args.mode in ["scaling", "all"]:       run_scaling()
    if args.mode in ["convergence", "all"]:   run_convergence()
    if args.mode in ["jacobi_tuning", "all"]: run_jacobi_tuning()

    print(f"\nDone. Results in {RESULTS_DIR}/")
