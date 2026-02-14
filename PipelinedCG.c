#include "solver_utils.h"
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
      printf("Usage: %s <n> [max_iter] [tol]\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  int max_iter = (argc >= 3) ? atoi(argv[2]) : 50000;
  double tol = (argc >= 4) ? atof(argv[3]) : 1e-6;

  GridContext ctx;
  SetupGrid(n, &ctx);

  int extRows = ctx.numRows + 2, extCols = ctx.numCols + 2;
  int localSize = ctx.numRows * ctx.numCols;
  int extSize = extRows * extCols;

  double *x = (double *)calloc(localSize, sizeof(double));
  double *b = (double *)malloc(localSize * sizeof(double));
  double *r = (double *)malloc(localSize * sizeof(double));
  double *u = (double *)malloc(localSize * sizeof(double));
  double *w = (double *)malloc(localSize * sizeof(double));
  double *p = (double *)malloc(localSize * sizeof(double));
  double *q = (double *)malloc(localSize * sizeof(double));
  double *m = (double *)malloc(localSize * sizeof(double));
  double *n_vec = (double *)malloc(localSize * sizeof(double));

  // Auxiliary ghost-zone buffers
  double *p_ext = (double *)calloc(extSize, sizeof(double));
  double *r_ext = (double *)calloc(extSize, sizeof(double));
  double *u_ext = (double *)calloc(extSize, sizeof(double));
  double *w_ext = (double *)calloc(extSize, sizeof(double));

  double *sendBuf = (double *)malloc((extRows - 2) * sizeof(double));
  double *recvBuf = (double *)malloc((extRows - 2) * sizeof(double));

  if (!x || !b || !r || !u || !w || !p || !q || !m || !n_vec) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++) {
      double xv = (ctx.I_START + i) * ctx.h;
      double yv = (ctx.J_START + j) * ctx.h;
      b[IDX(i, j, ctx.numCols)] =
          2 * ctx.h * ctx.h * (xv * (1 - xv) + yv * (1 - yv));
    }
  }

  // Step 0: x=0, r=b
  for (int i = 0; i < localSize; i++)
    r[i] = b[i];

  // u = A*r
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++)
      r_ext[IDX(i + 1, j + 1, extCols)] = r[IDX(i, j, ctx.numCols)];
  }
  exchangeBoundaryValues(r_ext, extRows, extCols, extCols, ctx.neighborProcs,
                         sendBuf, recvBuf);
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++) {
      int c = IDX(i + 1, j + 1, extCols);
      u[IDX(i, j, ctx.numCols)] = -1 * (r_ext[IDX(i, j + 1, extCols)] +
                                        r_ext[IDX(i + 2, j + 1, extCols)] +
                                        r_ext[IDX(i + 1, j, extCols)] +
                                        r_ext[IDX(i + 1, j + 2, extCols)]) +
                                  4 * r_ext[c];
    }
  }

  // w = A*u
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++)
      u_ext[IDX(i + 1, j + 1, extCols)] = u[IDX(i, j, ctx.numCols)];
  }
  exchangeBoundaryValues(u_ext, extRows, extCols, extCols, ctx.neighborProcs,
                         sendBuf, recvBuf);
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++) {
      int c = IDX(i + 1, j + 1, extCols);
      w[IDX(i, j, ctx.numCols)] = -1 * (u_ext[IDX(i, j + 1, extCols)] +
                                        u_ext[IDX(i + 2, j + 1, extCols)] +
                                        u_ext[IDX(i + 1, j, extCols)] +
                                        u_ext[IDX(i + 1, j + 2, extCols)]) +
                                  4 * u_ext[c];
    }
  }

  double local_dots[2] = {MatrixDotProduct(r, r, ctx.numRows, ctx.numCols,
                                           ctx.numCols, ctx.numCols, 0, 0),
                          MatrixDotProduct(r, u, ctx.numRows, ctx.numCols,
                                           ctx.numCols, ctx.numCols, 0, 0)};
  double global_dots[2]; // [0]=gamma, [1]=delta
  MPI_Allreduce(local_dots, global_dots, 2, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  double gamma = global_dots[0];
  double delta = global_dots[1];
  double alpha = gamma / delta;
  double beta = 0.0;

  // p=r, q=u, m=u, n=w
  for (int i = 0; i < localSize; i++) {
    p[i] = r[i];
    q[i] = u[i];
    m[i] = u[i];
    n_vec[i] = w[i];
  }

  double startTime = MPI_Wtime();
  int iter = 0;
  while (iter < max_iter) {
    if (tol > 0.0 && sqrt(gamma) < tol)
      break;

    // Step 1: Update x, r, u
    for (int i = 0; i < localSize; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * q[i];
      u[i] -= alpha * n_vec[i];
    }

    // Step 2: Iallreduce(r,r) and (r,u)
    local_dots[0] = MatrixDotProduct(r, r, ctx.numRows, ctx.numCols,
                                     ctx.numCols, ctx.numCols, 0, 0);
    local_dots[1] = MatrixDotProduct(r, u, ctx.numRows, ctx.numCols,
                                     ctx.numCols, ctx.numCols, 0, 0);
    MPI_Request req;
    MPI_Iallreduce(local_dots, global_dots, 2, MPI_DOUBLE, MPI_SUM,
                   MPI_COMM_WORLD, &req);

    // Step 3: Overlap SpMV: w = A*u
    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++)
        u_ext[IDX(i + 1, j + 1, extCols)] = u[IDX(i, j, ctx.numCols)];
    }
    exchangeBoundaryValues(u_ext, extRows, extCols, extCols, ctx.neighborProcs,
                           sendBuf, recvBuf);
    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++) {
        int c = IDX(i + 1, j + 1, extCols);
        w[IDX(i, j, ctx.numCols)] = -1 * (u_ext[IDX(i, j + 1, extCols)] +
                                          u_ext[IDX(i + 2, j + 1, extCols)] +
                                          u_ext[IDX(i + 1, j, extCols)] +
                                          u_ext[IDX(i + 1, j + 2, extCols)]) +
                                    4 * u_ext[c];
      }
    }

    MPI_Wait(&req, MPI_STATUS_IGNORE);

    double gamma_new = global_dots[0];
    double delta_new = global_dots[1];

    beta = gamma_new / gamma;
    gamma = gamma_new;
    delta = delta_new;

    alpha = gamma / (delta - (beta / alpha) * gamma);

    // Step 4: Update p, q, n (m is used for q update in some variants, here q
    // update is direct)
    for (int i = 0; i < localSize; i++) {
      p[i] = r[i] + beta * p[i];
      q[i] = u[i] + beta * q[i];
      n_vec[i] = w[i] + beta * n_vec[i];
    }

    iter++;
  }

  double exe_time = MPI_Wtime() - startTime;
  double max_exe_time;
  MPI_Reduce(&exe_time, &max_exe_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (myid == 0) {
    printf("Pipelined CG - Iterations: %d, Time: %f\n", iter, max_exe_time);
  }

  free(x);
  free(b);
  free(r);
  free(u);
  free(w);
  free(p);
  free(q);
  free(m);
  free(n_vec);
  free(p_ext);
  free(r_ext);
  free(u_ext);
  free(w_ext);
  free(sendBuf);
  free(recvBuf);
  MPI_Finalize();
  return 0;
}
