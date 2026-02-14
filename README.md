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

- **Weak Scaling**: Achieves ~39% efficiency at 25 processes on the UPPMAX cluster (512x512 local grid per process)
- **Strong Scaling**: Achieves **13.4x speedup** on 25 cores (UPPMAX cluster, n=2048)
- **Algorithm Efficiency**: Demonstrates optimal $O(n)$ iteration scaling characteristic of CG methods
- **Algorithmic Extensions**: Block-Jacobi preconditioning reduces iteration count by ~30% with zero additional communication overhead.
- **Numerical Stability**: High-precision verification with identical results across different MPI implementations and scale factors
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
*   OpenMPI / MPICH

```bash
# Compilation
make
# Run with $P$ processes (must be a perfect square, e.g., 4, 16, 64):
# mpirun -n procs ./CG grid_size max_iter tol

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
