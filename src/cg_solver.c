#include "comm.h"
#include "linalg.h"
#include "precond.h"
#include "solver_structs.h"
#include "solver_utils.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  int myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (argc < 2) {
    if (myid == 0)
      printf("Usage: %s <n> [max_iter] [tol] [jacobi_iters]\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  int max_iter = (argc >= 3) ? atoi(argv[2]) : DEFAULT_MAX_ITER;
  double tol = (argc >= 4) ? atof(argv[3]) : 0.0;
  int jacobi_iters = (argc >= 5) ? atoi(argv[4]) : 5;

  // Solver Context Initialization
  SolverContext ctx;
  memset(&ctx, 0, sizeof(SolverContext));

  if (SetupGrid(n, &ctx.grid) != 0) {
    MPI_Finalize();
    return 1;
  }

  ctx.tol = tol;
  ctx.max_iter = max_iter;
  ctx.jacobi_iters = jacobi_iters;

  // Auto-detect preconditioner type based on binary name or arg?
  // For now, let's default to Jacobi since this is "Profling Sandbox".
  // Or check if argv[0] contains "PCG"
  if (strstr(argv[0], "PCG") != NULL) {
    ctx.precond_type = PRECOND_JACOBI;
  } else {
    ctx.precond_type = PRECOND_NONE;
  }

  // Vector Allocation
  Vector *b = create_vector(ctx.grid.numRows, ctx.grid.numCols);
  Vector *x = create_vector(ctx.grid.numRows, ctx.grid.numCols); // solution u
  Vector *r = create_vector(ctx.grid.numRows, ctx.grid.numCols); // residual g
  Vector *p = create_vector(ctx.grid.numRows, ctx.grid.numCols); // direction d
  Vector *q = create_vector(ctx.grid.numRows, ctx.grid.numCols); // A * p
  Vector *z =
      create_vector(ctx.grid.numRows, ctx.grid.numCols); // preconditioned r

  // Initialize RHS b
  // (Math-as-Code goal: hide raw loops, but initialization is specific)
  for (int i = 0; i < ctx.grid.numRows; i++) {
    for (int j = 0; j < ctx.grid.numCols; j++) {
      double xx = (ctx.grid.I_START + i) * ctx.grid.h;
      double yy = (ctx.grid.J_START + j) * ctx.grid.h;
      int idx = IDX(i + 1, j + 1, b->stride);
      b->data[idx] =
          2 * ctx.grid.h * ctx.grid.h * (xx * (1 - xx) + yy * (1 - yy));
    }
  }

  double start_time = MPI_Wtime();

  // --- Initialization Phase ---
  // x = 0
  vec_set(x, 0.0);

  // r = b - A*x (since x=0, r=b)
  copy_vector(r, b);

  // z = M^{-1} r
  apply_preconditioner(z, r, &ctx);

  // p = z
  copy_vector(p, z);

  double rho = dot_product_global(r, z, &ctx);
  double rho_old = rho;

  // --- Main Solver Loop ---
  int iter = 0;
  while (iter < max_iter) {
    // 1. Halo Exchange (Communication)
    halo_exchange(p, &ctx);

    // 2. Matrix-Vector Product q = A * p (Compute)
    apply_stencil(q, p, &ctx);

    // 3. Global Dot Product (Communication/Reduction)
    double pAq = dot_product_global(p, q, &ctx);
    double alpha = rho / pAq;

    // 4. Update Solution & Residual (Memory Bandwidth)
    vec_axpy(x, p, alpha, &ctx);  // x = x + alpha * p
    vec_axpy(r, q, -alpha, &ctx); // r = r - alpha * q

    // 5. Convergence Check
    // Using un-preconditioned residual for check? Standard practice varies.
    // Let's stick to ||r|| < tol
    double r_norm2 = dot_product_global(r, r, &ctx);
    if (tol > 0.0 && sqrt(r_norm2) < tol) {
      iter++;
      break;
    }

    // 6. Preconditioning (Compute/Memory)
    apply_preconditioner(z, r, &ctx);

    // 7. Update Search Direction
    double rho_new = dot_product_global(r, z, &ctx);
    double beta = rho_new / rho;

    vec_xpay(p, z, beta, &ctx); // p = z + beta * p

    rho = rho_new;
    iter++;
  }

  // --- Final Reporting ---
  double end_time = MPI_Wtime();
  ctx.prof.time_total = end_time - start_time;

  // Aggregate max time across procs for reporting?
  // Profiler struct doesn't have max-reduction logic yet.
  // For "Sandbox" purposes, Rank 0 reporting is often enough,
  // assuming reasonable load balance.

  print_profiling_results(&ctx);

  if (myid == 0) {
    printf("Iterations: %d\n", iter);
  }

  // Cleanup
  free_vector(b);
  free_vector(x);
  free_vector(r);
  free_vector(p);
  free_vector(q);
  free_vector(z);

  MPI_Finalize();
  return 0;
}
