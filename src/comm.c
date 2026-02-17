#include "comm.h"
#include "linalg.h" // for vec_dot_local
#include "utils.h"
#include <mpi.h>

void halo_exchange(Vector *v, SolverContext *ctx) {
  TIC(&ctx->prof.time_halo);

  // Reuse existing logic from exchangeBoundaryValues but adapt to Vector struct
  // We need buffers for non-contiguous packing (left/right boundaries)
  // For now, let's alloc/free temp buffers here, or we could add them to
  // SolverContext to avoid malloc overhead. Given the "alloc inside loop"
  // warning in previous conversations, context-based buffers would be better.
  // However, for this step, let's implement using local malloc for simplicity,
  // then optimize.

  // Actually, solver_utils.c had `exchangeBoundaryValues`. We can either call
  // that or reimplement it here cleanly. Reimplementing is better for
  // "Math-as-Code" purity.

  int rows = v->num_rows;
  int cols = v->num_cols;
  int stride = v->stride;
  double *d = v->data;
  int *neighbors = ctx->grid.neighborProcs;

  // Buffers for left/right
  int inner_rows = rows;
  double *sendBuf = (double *)malloc(inner_rows * sizeof(double));
  double *recvBuf = (double *)malloc(inner_rows * sizeof(double));

  // UP (Send 1st row data, Recv into 0th ghost row)
  if (neighbors[0] != -1) {
    MPI_Sendrecv(&d[IDX(1, 0, stride)], cols, MPI_DOUBLE, neighbors[0], 0,
                 &d[IDX(0, 0, stride)], cols, MPI_DOUBLE, neighbors[0], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // DOWN (Send last row data, Recv into last ghost row)
  // Last row data index: rows (since 1-based start)
  // Last ghost row index: rows + 1
  if (neighbors[1] != -1) {
    MPI_Sendrecv(&d[IDX(rows, 0, stride)], cols, MPI_DOUBLE, neighbors[1], 0,
                 &d[IDX(rows + 1, 0, stride)], cols, MPI_DOUBLE, neighbors[1],
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // LEFT (Send 1st col, Recv into 0th ghost col)
  if (neighbors[2] != -1) {
    for (int i = 0; i < inner_rows; i++)
      sendBuf[i] = d[IDX(i + 1, 1, stride)];

    MPI_Sendrecv(sendBuf, inner_rows, MPI_DOUBLE, neighbors[2], 0, recvBuf,
                 inner_rows, MPI_DOUBLE, neighbors[2], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (int i = 0; i < inner_rows; i++)
      d[IDX(i + 1, 0, stride)] = recvBuf[i];
  }

  // RIGHT (Send last col, Recv into last ghost col)
  if (neighbors[3] != -1) {
    // Last col index: cols
    for (int i = 0; i < inner_rows; i++)
      sendBuf[i] = d[IDX(i + 1, cols, stride)];

    MPI_Sendrecv(sendBuf, inner_rows, MPI_DOUBLE, neighbors[3], 0, recvBuf,
                 inner_rows, MPI_DOUBLE, neighbors[3], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (int i = 0; i < inner_rows; i++)
      d[IDX(i + 1, cols + 1, stride)] = recvBuf[i];
  }

  free(sendBuf);
  free(recvBuf);

  TOC(&ctx->prof.time_halo, &ctx->prof.time_halo);
}

double dot_product_global(Vector *a, Vector *b, SolverContext *ctx) {
  // 1. Local computation
  // We do NOT want to profile local computation as 'allreduce' time.
  // But typically dot product is considered part of the "scalar ops" bucket.
  // Let's keep local separate or just include it? The request said "internal
  // automatic record". Usually dot product is memory-bound (local) +
  // latency-bound (global). Let's use time_allreduce strictly for the MPI part.

  double local_sum = vec_dot_local(a, b);
  double global_sum = 0.0;

  TIC(&ctx->prof.time_allreduce);
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  TOC(&ctx->prof.time_allreduce, &ctx->prof.time_allreduce);

  return global_sum;
}
