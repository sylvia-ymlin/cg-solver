#include "utils.h"
#include <string.h>

Vector *create_vector(int num_rows, int num_cols) {
  Vector *vec = (Vector *)malloc(sizeof(Vector));
  if (!vec)
    return NULL;

  vec->num_rows = num_rows;
  vec->num_cols = num_cols;

  // Add halo regions (1 row/col on each side)
  // Stride is usually logical cols + 2 (left + right halo)
  int ext_rows = num_rows + 2;
  int ext_cols = num_cols + 2;
  vec->stride = ext_cols;

  // Use calloc to initialize to zero (helpful for ghost cells)
  vec->data = (double *)calloc(ext_rows * ext_cols, sizeof(double));
  if (!vec->data) {
    free(vec);
    return NULL;
  }

  return vec;
}

void free_vector(Vector *vec) {
  if (vec) {
    if (vec->data)
      free(vec->data);
    free(vec);
  }
}

void copy_vector(Vector *dst, Vector *src) {
  // We copy the entire buffer including ghost cells
  // Size = stride * (num_rows + 2)
  // Note: stride = num_cols + 2
  int total_size = src->stride * (src->num_rows + 2);
  memcpy(dst->data, src->data, total_size * sizeof(double));

  // Copy metadata just in case
  dst->num_rows = src->num_rows;
  dst->num_cols = src->num_cols;
  dst->stride = src->stride;
}

void print_profiling_results(SolverContext *ctx) {
  int myid = ctx->grid.myid;

  // Only rank 0 prints the report
  if (myid != 0)
    return;

  printf("\n=== Profiling Report (Rank 0) ===\n");
  printf("Total Time:      %f s\n", ctx->prof.time_total);
  printf("--------------------------------\n");
  printf("Compute (Stencil): %f s\n", ctx->prof.time_stencil);
  printf("Memory (AXPY):     %f s\n", ctx->prof.time_axpy);
  printf("Comm (Halo):       %f s\n", ctx->prof.time_halo);
  printf("Comm (Allreduce):  %f s\n", ctx->prof.time_allreduce);
  printf("Preconditioner:    %f s\n", ctx->prof.time_precond);
  printf("--------------------------------\n");
}
