#ifndef SOLVER_STRUCTS_H
#define SOLVER_STRUCTS_H

#include "solver_utils.h" // For GridContext

// Vector abstraction to bundle data with dimensions
typedef struct {
  double *data; // Pointer to local data (including ghost cells)
  int num_rows; // Logic rows (excluding ghost)
  int num_cols; // Logic cols (excluding ghost)
  int stride;   // Physical stride (usually num_cols + 2)
} Vector;

// Explicit profiler to capture time spent in different phases
typedef struct {
  double time_halo;      // Communication: Boundary exchange
  double time_stencil;   // Compute: Matrix-vector product (A*p)
  double time_allreduce; // Communication: Global reduction (dot product)
  double time_axpy;      // Memory: Vector updates (axpy, xpay)
  double time_precond;   // Compute/Memory: Preconditioning application
  double time_total;     // Total solve time
} Profiler;

// Supported preconditioner types
typedef enum { PRECOND_NONE, PRECOND_JACOBI } PrecondType;

// Aggregated context for the solver
typedef struct {
  GridContext grid;         // MPI and Grid topology info
  Profiler prof;            // Performance counters
  double tol;               // Convergence tolerance
  int max_iter;             // Maximum iterations
  PrecondType precond_type; // Selected preconditioner
  int jacobi_iters;         // Iterations for Block-Jacobi (if used)
} SolverContext;

#endif // SOLVER_STRUCTS_H
