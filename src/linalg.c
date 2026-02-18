#include "linalg.h"
#include "solver_utils.h" // For IDX macro
#include "utils.h"

void vec_axpy(Vector *y, Vector *x, double alpha, SolverContext *ctx) {
  if (!y || !x)
    return;

  TIC(&ctx->prof.time_axpy);

  int size = (y->num_rows + 2) * y->stride;

  // Naively update everything including ghost cells (or just internal?)
  // Typically updates strictly logical domain, but since ghost cells are
  // overwritten by halo exchange, we can just iterate over logical domain.
  // For simplicity and bandwidth measurement, we iterate over the whole valid
  // grid.

  // Let's stick to logical domain for correctness to avoid propagating garbage
  int rows = y->num_rows;
  int cols = y->num_cols;
  int stride = y->stride; // stride is usually cols + 2

  double *py = y->data;
  double *px = x->data;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // logical (i,j) maps to data[IDX(i+1, j+1, stride)]
      int idx = IDX(i + 1, j + 1, stride);
      py[idx] += alpha * px[idx];
    }
  }

  TOC(&ctx->prof.time_axpy, &ctx->prof.time_axpy);
}

void vec_xpay(Vector *y, Vector *x, double beta, SolverContext *ctx) {
  if (!y || !x)
    return;

  TIC(&ctx->prof.time_axpy);

  int rows = y->num_rows;
  int cols = y->num_cols;
  int stride = y->stride;

  double *py = y->data;
  double *px = x->data;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = IDX(i + 1, j + 1, stride);
      py[idx] = px[idx] + beta * py[idx];
    }
  }

  TOC(&ctx->prof.time_axpy, &ctx->prof.time_axpy);
}

void apply_stencil(Vector *q, Vector *p, SolverContext *ctx) {
  if (!q || !p)
    return;

  TIC(&ctx->prof.time_stencil);

  int rows = q->num_rows;
  int cols = q->num_cols;
  int stride = q->stride;
  double *pq = q->data;
  double *pp = p->data;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // Center is at (i+1, j+1) due to ghost cells
      int c = IDX(i + 1, j + 1, stride);
      int up = IDX(i, j + 1, stride);
      int down = IDX(i + 2, j + 1, stride);
      int left = IDX(i + 1, j, stride);
      int right = IDX(i + 1, j + 2, stride);

      // 5-point stencil: q = A*p where A is the discrete Laplacian
      pq[c] = 4 * pp[c] - (pp[up] + pp[down] + pp[left] + pp[right]);
    }
  }

  TOC(&ctx->prof.time_stencil, &ctx->prof.time_stencil);
}

void vec_set(Vector *v, double val) {
  if (!v)
    return;
  int size = v->stride * (v->num_rows + 2);
  for (int i = 0; i < size; i++) {
    v->data[i] = val;
  }
}

double vec_dot_local(Vector *a, Vector *b) {
  double sum = 0.0;
  int rows = a->num_rows;
  int cols = a->num_cols;
  int stride = a->stride;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = IDX(i + 1, j + 1, stride);
      sum += a->data[idx] * b->data[idx];
    }
  }
  return sum;
}
