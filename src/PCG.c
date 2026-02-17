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
      printf("Usage: %s <n> [max_iter] [tol] [jacobi_iters]\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  int max_iter = (argc >= 3) ? atoi(argv[2]) : DEFAULT_MAX_ITER;
  double tol = (argc >= 4) ? atof(argv[3]) : 0.0;
  int jacobi_iters = (argc >= 5) ? atoi(argv[4]) : 5;

  GridContext ctx;
  if (SetupGrid(n, &ctx) != 0) {
    MPI_Finalize();
    return 1;
  }

  // Memory Allocation
  double *b = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));
  double *u = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));
  double *g = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));
  double *q = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));
  double *z = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));
  int extRows = ctx.numRows + 2, extCols = ctx.numCols + 2;
  double *d = (double *)calloc(extRows * extCols, sizeof(double));
  double *sendBuf = (double *)malloc((extRows - 2) * sizeof(double));
  double *recvBuf = (double *)malloc((extRows - 2) * sizeof(double));
  // Auxiliary vector for Preconditioning
  double *z_new = (double *)malloc(ctx.numRows * ctx.numCols * sizeof(double));

  // Initialize b
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++) {
      double x = (ctx.I_START + i) * ctx.h;
      double y = (ctx.J_START + j) * ctx.h;
      b[IDX(i, j, ctx.numCols)] =
          2 * ctx.h * ctx.h * (x * (1 - x) + y * (1 - y));
    }
  }

  double startTime = MPI_Wtime();

  // 1. u = 0, r = b
  for (int i = 0; i < ctx.numRows * ctx.numCols; i++) {
    u[i] = 0;
    g[i] = b[i];
  }

  // Initial Preconditioning: Mz = r (Jacobi)
  for (int i = 0; i < ctx.numRows * ctx.numCols; i++)
    z[i] = g[i] / 4.0;

  // Initial Search Direction: d = z
  for (int i = 0; i < ctx.numRows; i++) {
    for (int j = 0; j < ctx.numCols; j++) {
      d[IDX(i + 1, j + 1, extCols)] = z[IDX(i, j, ctx.numCols)];
    }
  }

  double rho0 = MatrixDotProduct(g, z, ctx.numRows, ctx.numCols, ctx.numCols,
                                 ctx.numCols, 0, 0);
  MPI_Allreduce(MPI_IN_PLACE, &rho0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int iter = 0;
  while (iter < max_iter) {
    exchangeBoundaryValues(d, extRows, extCols, extCols, ctx.neighborProcs,
                           sendBuf, recvBuf);

    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++) {
        int c = IDX(i + 1, j + 1, extCols);
        q[IDX(i, j, ctx.numCols)] =
            -1 * (d[IDX(i, j + 1, extCols)] + d[IDX(i + 2, j + 1, extCols)] +
                  d[IDX(i + 1, j, extCols)] + d[IDX(i + 1, j + 2, extCols)]) +
            4 * d[c];
      }
    }

    double denom = MatrixDotProduct(d, q, ctx.numRows, ctx.numCols, extCols,
                                    ctx.numCols, 1, 1);
    MPI_Allreduce(MPI_IN_PLACE, &denom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double alpha = rho0 / denom;

    // u = u + alpha * d
    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++) {
        u[IDX(i, j, ctx.numCols)] += alpha * d[IDX(i + 1, j + 1, extCols)];
      }
    }

    // g = g - alpha * q
    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++) {
        g[IDX(i, j, ctx.numCols)] -= alpha * q[IDX(i, j, ctx.numCols)];
      }
    }

    double r_norm_sq = MatrixDotProduct(g, g, ctx.numRows, ctx.numCols,
                                        ctx.numCols, ctx.numCols, 0, 0);
    MPI_Allreduce(MPI_IN_PLACE, &r_norm_sq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    if (tol > 0.0 && sqrt(r_norm_sq) < tol) {
      iter++;
      break;
    }

    // Mz = r (Block-Jacobi with implicit 5 iterations or specified)
    for (int i = 0; i < ctx.numRows * ctx.numCols; i++)
      z[i] = 0.0;
    for (int it = 0; it < jacobi_iters; it++) {
      for (int i = 0; i < ctx.numRows; i++) {
        for (int j = 0; j < ctx.numCols; j++) {
          double up = (i > 0) ? z[IDX(i - 1, j, ctx.numCols)] : 0.0;
          double down =
              (i < ctx.numRows - 1) ? z[IDX(i + 1, j, ctx.numCols)] : 0.0;
          double left = (j > 0) ? z[IDX(i, j - 1, ctx.numCols)] : 0.0;
          double right =
              (j < ctx.numCols - 1) ? z[IDX(i, j + 1, ctx.numCols)] : 0.0;
          z_new[IDX(i, j, ctx.numCols)] =
              (g[IDX(i, j, ctx.numCols)] + up + down + left + right) / 4.0;
        }
      }
      memcpy(z, z_new, ctx.numRows * ctx.numCols * sizeof(double));
    }

    double rho1 = MatrixDotProduct(g, z, ctx.numRows, ctx.numCols, ctx.numCols,
                                   ctx.numCols, 0, 0);
    MPI_Allreduce(MPI_IN_PLACE, &rho1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double beta = rho1 / rho0;
    // d = z + beta * d
    for (int i = 0; i < ctx.numRows; i++) {
      for (int j = 0; j < ctx.numCols; j++) {
        d[IDX(i + 1, j + 1, extCols)] =
            z[IDX(i, j, ctx.numCols)] + beta * d[IDX(i + 1, j + 1, extCols)];
      }
    }
    rho0 = rho1;
    iter++;
  }

  double exe_time = MPI_Wtime() - startTime;
  double max_exe_time;
  MPI_Reduce(&exe_time, &max_exe_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (myid == 0) {
    printf("Preconditioned CG - Iterations: %d, Time: %f\n", iter,
           max_exe_time);
  }

  free(z_new);
  free(z);
  free(sendBuf);
  free(recvBuf);
  free(d);
  free(b);
  free(u);
  free(g);
  free(q);
  MPI_Finalize();
  return 0;
}
