#include "precond.h"
#include "solver_utils.h" // For IDX
#include "string.h"
#include "utils.h"

void apply_preconditioner(Vector *z, Vector *r, SolverContext *ctx) {
  if (!z || !r)
    return;

  if (ctx->precond_type == PRECOND_NONE) {
    // z = r (Identity preconditioner)
    // This is memory-bound copy
    TIC(&ctx->prof.time_precond);
    copy_vector(z, r);
    TOC(&ctx->prof.time_precond, &ctx->prof.time_precond);
    return;
  }

  if (ctx->precond_type == PRECOND_JACOBI) {
    TIC(&ctx->prof.time_precond);

    int rows = z->num_rows;
    int cols = z->num_cols;
    int stride = z->stride;
    int iters = ctx->jacobi_iters;

    double *pz = z->data;
    double *pr = r->data;

    // Reuse r as initial guess directly? Or zero guess?
    // Standard PCG often uses z=0 as initial guess for Mz=r solve
    // Initial guess z = r/4.0 in original PCG
    for (int i = 0; i < stride * (rows + 2); i++) {
      // Initial guess: diagonal scaling
      // A's diagonal is roughly 4 (from 4*u_ij)
      // So D^{-1} r roughly r / 4.0
      pz[i] = pr[i] / 4.0;
    }

    // We need a temp buffer for Jacobi iteration
    // In original PCG, z_new was used.
    // We should probably allocate z_new here or attached to context.
    // For strict profiling, let's alloc/free here (overhead included in precond
    // time) or prioritize clean code.

    Vector *z_new = create_vector(rows, cols);

    for (int k = 0; k < iters; k++) {
      double *pzn = z_new->data;

      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int c = IDX(i + 1, j + 1, stride);
          int up = IDX(i, j + 1, stride);
          int down = IDX(i + 2, j + 1, stride);
          int left = IDX(i + 1, j, stride);
          int right = IDX(i + 1, j + 2, stride);

          // Jacobi step: z_new = D^{-1} (r - (L+U)z_old)
          // A = 4I - (L+U)
          // r = Az = (4I - (L+U))z
          // r + (L+U)z = 4z
          // z_new = (r + neighbors_sum) / 4.0

          double neighbors_sum = pz[up] + pz[down] + pz[left] + pz[right];
          pzn[c] = (pr[c] + neighbors_sum) / 4.0;
        }
      }

      // Swap pointers or copy
      // Copy for simplicity with struct structure
      // Or just swap data pointers? Mapping back to struct is safer with copy
      // given existing copy_vector implementation
      // copy_vector(z, z_new); // This copies ghost cells too?
      // Jacobi needs ghost cells update?
      // Wait, Block Jacobi usually implies local solve, ignoring neighbor
      // procs? "Block-Jacobi preconditioner... solved locally on each
      // processor" So we do NOT exchange boundaries here.

      // The boundaries (ghost cells) of z are 0?
      // In original PCG code:
      // "double up = (i > 0) ? z[IDX(i-1, ...)] : 0.0"
      // It explicitly handled local boundaries as 0 (Dirichlet on block
      // boundary). Our Vector has ghost cells initialized to 0. So accessing
      // pz[up] where up is a ghost cell returns 0. Correct.

      // We need to copy z_new back to z
      // Optimizing: swap pointers?
      // Since we allocated z_new, we can just swap data ptrs
      double *tmp = z->data;
      z->data = z_new->data;
      z_new->data = tmp;
    }

    free_vector(z_new);

    TOC(&ctx->prof.time_precond, &ctx->prof.time_precond);
    return;
  }
}
