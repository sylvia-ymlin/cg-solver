# MPI Parallel Conjugate Gradient Solver

![Language](https://img.shields.io/badge/Language-C99-blue.svg)
![Parallelism](https://img.shields.io/badge/Parallelism-MPI-green.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-Conjugate%20Gradient-orange.svg)

An MPI-parallelized CG solver for self-adjoint elliptic PDEs (2D Poisson), featuring matrix-free stencil operations, 2D domain decomposition, and Block-Jacobi preconditioning.

> **Note**: For detailed mathematical derivation, complexity analysis, and extensive benchmarks, see the **[Project Report](docs/REPORT.md)**.

## 🚀 Quick Start

### Prerequisites
*   GCC / Clang
*   OpenMPI (`brew install open-mpi`)

### Build & Run
```bash
# 1. Compile
make

# 2. Run Single Instance (e.g., 1024x1024 grid, 4 processes)
mpirun -n 4 ./PCG 1024 1000 1e-6 10
# Arguments: <n> <max_iter> <tol> [jacobi_steps]

# 3. Run Validation
make && bash scripts/validate.sh
```

### Reproduce Benchmarks
The project includes a unified script to generate all performance plots:
```bash
# Run all scaling, convergence, and comparison benchmarks
python3 scripts/benchmark.py all
# View the generated report
open docs/REPORT.md
```

## 📊 Key Results

Experiments on a local 8-core Mac Workstation demonstrate:

*   **Memory Wall**: Strong scaling plateaus at $p=9$ despite available cores, confirming memory bandwidth saturation typical of stencil codes on unified memory architectures.
*   **Numerical Optimization**: **Block-Jacobi Preconditioning** reduces iteration count by **~30%**, though total time benefits are limited significantly by memory overhead on this hardware.
*   **Optimal Tuning**: A "sweet spot" of **10 preconditioning steps** minimizes total time to solution.
*   **Implementation Efficiency**: Matrix-free approach yields **10x memory savings** compared to explicit sparse matrix storage.

## ✨ Features

*   **2D Domain Decomposition**: Minimizes communication surface area ($4n/\sqrt{p}$).
*   **Matrix-Free**: High arithmetic intensity, low memory footprint.
*   **Communication Hiding**: Design ready for overlap (though limited by local hardware latency).
*   **System & Numerical Hybrid**: Optimizes both the algorithm ($O(n)$ iterations) and the implementation ($T_{iter}$).

