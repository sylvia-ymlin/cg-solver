# MPI Parallel Conjugate Gradient Solver

![Language](https://img.shields.io/badge/Language-C99-blue.svg)
![Parallelism](https://img.shields.io/badge/Parallelism-MPI-green.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-Conjugate%20Gradient-orange.svg)

An MPI-parallelized CG solver for self-adjoint elliptic PDEs, validated against a theoretical performance model, with extensions into preconditioning and communication hiding.

For detailed performance analysis, benchmarks, and technical discussion, see the full [Project Report](docs/REPORT.md).

## Implementation Features

*   **2D Domain Decomposition**: Splits computational grid into process-local blocks for optimal weak scaling
*   **Efficient Halo Exchange**: Uses `MPI_Sendrecv` for boundary synchronization with deadlock avoidance
*   **Matrix-Free Implementation**: Stencil-based approach for 10x memory savings over explicit sparse matrices
*   **Parallel Reduction**: Optimized global dot-products using `MPI_Allreduce`

## Performance Results

- **Weak Scaling**: Analysis confirms **Memory Bandwidth Saturation** determines performance on local hardware. At $p=9$, efficiency drops to ~24% due to core oversubscription and shared memory contention.
- **Strong Scaling**: Achieves **2x speedup** on 4 cores (Mac M1/Intel), clearly demonstrating the **Memory Wall** bottleneck typical of stencil computations on unified memory architectures.
- **Algorithm Efficiency**: Demonstrates optimal $O(n)$ iteration scaling characteristic of CG methods
- **Algorithmic Extensions**: Block-Jacobi preconditioning reduces iteration count by ~30%.
- **Numerical Stability**: High-precision verification with identical results across different MPI implementations.
- **Memory Optimization**: 10x memory savings through matrix-free stencil implementation

## Conjugate Gradient Algorithm

### Mathematical Formulation
The solver addresses general SPD linear systems of the form:
$$ Ax = b $$
In this implementation, $A$ represents a 5-point Laplacian stencil, enabling a **matrix-free** approach that avoids the memory overhead of explicit sparse matrix storage.

**Convergence Criteria**:
The solver iterates until the residual norm satisfies:
$$ \|r_k\|_2 = \|b - Ax_k\|_2 < \epsilon $$

### Algorithm Steps
For the $k$-th iteration:
1.  Calculate step size:
    $$ \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k} $$
2.  Update solution and residual:
    $$ x_{k+1} = x_k + \alpha_k p_k $$
    $$ r_{k+1} = r_k - \alpha_k A p_k $$
3.  Calculate search direction correction:
    $$ \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k} $$
4.  Update search direction:
    $$ p_{k+1} = r_{k+1} + \beta_k p_k $$

### Algorithm Features
The CG method is particularly well-suited for:
- **Symmetric Positive Definite (SPD) matrices**: Guaranteed convergence with optimal Krylov subspace properties
- **Memory efficiency**: Minimal storage requirements compared to direct solvers
- **Short recurrence**: Only one matrix-vector product per iteration
- **Monotonic convergence**: Residual decreases monotonically at each iteration
- **Optimal convergence**: Among the fastest converging Krylov methods for SPD systems

## Build & Run

### Prerequisites
*   GCC / Clang
*   OpenMPI (`brew install open-mpi`)

```bash
# Compilation
make
# Run with $P$ processes (must be a perfect square, e.g., 4, 16, 64):
# Usage
mpirun -n <np> ./PCG <n> [max_iter] [tol] [jacobi_iters]
# Example: 10 preconditioning steps
mpirun -n 4 ./PCG 1024 1000 1e-6 10

# Fixed iterations (timing mode, default 200 iters):
mpirun -n 16 ./CG 1000

# Run to convergence (accuracy mode):
mpirun -n 16 ./CG 1000 50000 1e-6
```
*   `procs`: Number of MPI ranks (must be a perfect square)
*   `grid_size`: Number of intervals along one axis (total unknowns = $n^2$)
*   `max_iter` (optional): Maximum iterations (default: 200)
*   `tol` (optional): Convergence tolerance on $\|r\|_2$ (default: 0, meaning no check)

## Quick Validation

```bash
# Build and validate
make && bash scripts/validate.sh

# Expected output:
mpirun -np 1 ./CG 256 50000 1e-6
Iterations: 281
Converged: yes
Residual: 9.8156129832e-07
L2_Error: 6.6555733901e-08
```
```

## Reproducing Results

The project includes a unified benchmark script `scripts/benchmark.py` to generate all performance data and plots.

```bash
# 1. Compile
make

# 2. Run Benchmarks
# Options: scaling (default), convergence, comparison, or all
python3 scripts/benchmark.py all

# 3. View Report
open docs/REPORT.md
```
