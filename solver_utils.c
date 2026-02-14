#include "solver_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void SetupGrid(int n, GridContext *ctx) {
  int myid, numprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  ctx->myid = myid;
  ctx->numprocs = numprocs;
  ctx->n = n;
  ctx->h = 1.0 / (n + 1);

  int gridDim = (int)sqrt(numprocs);
  if (gridDim * gridDim != numprocs) {
    if (myid == 0)
      printf("Error: Number of processes must be a perfect square.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ctx->gridDim = gridDim;

  if (n < gridDim) {
    if (myid == 0)
      printf("Error: n (%d) must be >= sqrt(numprocs) (%d).\n", n, gridDim);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ctx->row = myid / gridDim;
  ctx->col = myid % gridDim;
  ctx->neighborProcs[0] = (myid - gridDim >= 0) ? myid - gridDim : -1;
  ctx->neighborProcs[1] = (myid + gridDim < numprocs) ? myid + gridDim : -1;
  ctx->neighborProcs[2] = (myid % gridDim != 0) ? myid - 1 : -1;
  ctx->neighborProcs[3] = (myid % gridDim != gridDim - 1) ? myid + 1 : -1;

  int blockSize = n / gridDim;
  int residual = n % gridDim;
  ctx->numRows = blockSize;
  ctx->numCols = blockSize;

  if (ctx->row < residual) {
    ctx->numRows++;
    ctx->I_START = ctx->row * (blockSize + 1) + 1;
  } else {
    ctx->I_START =
        residual * (blockSize + 1) + (ctx->row - residual) * blockSize + 1;
  }

  if (ctx->col < residual) {
    ctx->numCols++;
    ctx->J_START = ctx->col * (blockSize + 1) + 1;
  } else {
    ctx->J_START =
        residual * (blockSize + 1) + (ctx->col - residual) * blockSize + 1;
  }
}

double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col) {
  double sum = 0.0;
  int startA = IDX(offsetA_row, offsetA_col, strideA);
  int startB = 0;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sum += A[startA + i * strideA + j] * B[startB + i * strideB + j];
    }
  }
  return sum;
}

void MatrixAdd(double *A, double *B, double a, double b, double *C, int rows,
               int cols, int strideA, int strideB, int strideC, int offAr,
               int offAc, int offBr, int offBc, int offCr, int offCc) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double valA = A[IDX(i + offAr, j + offAc, strideA)];
      double valB = B[IDX(i + offBr, j + offBc, strideB)];
      C[IDX(i + offCr, j + offCc, strideC)] = a * valA + b * valB;
    }
  }
}

void exchangeBoundaryValues(double *d, int numRows, int numCols, int stride,
                            int *neighborProcs, double *sendBuf,
                            double *recvBuf) {
  if (neighborProcs[0] != -1) {
    MPI_Sendrecv(&d[IDX(1, 0, stride)], numCols, MPI_DOUBLE, neighborProcs[0],
                 0, &d[IDX(0, 0, stride)], numCols, MPI_DOUBLE,
                 neighborProcs[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (neighborProcs[1] != -1) {
    MPI_Sendrecv(&d[IDX(numRows - 2, 0, stride)], numCols, MPI_DOUBLE,
                 neighborProcs[1], 0, &d[IDX(numRows - 1, 0, stride)], numCols,
                 MPI_DOUBLE, neighborProcs[1], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }
  int innerRows_local = numRows - 2;
  if (neighborProcs[2] != -1) {
    for (int i = 0; i < innerRows_local; i++)
      sendBuf[i] = d[IDX(i + 1, 1, stride)];
    MPI_Sendrecv(sendBuf, innerRows_local, MPI_DOUBLE, neighborProcs[2], 0,
                 recvBuf, innerRows_local, MPI_DOUBLE, neighborProcs[2], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < innerRows_local; i++)
      d[IDX(i + 1, 0, stride)] = recvBuf[i];
  }
  if (neighborProcs[3] != -1) {
    for (int i = 0; i < innerRows_local; i++)
      sendBuf[i] = d[IDX(i + 1, numCols - 2, stride)];
    MPI_Sendrecv(sendBuf, innerRows_local, MPI_DOUBLE, neighborProcs[3], 0,
                 recvBuf, innerRows_local, MPI_DOUBLE, neighborProcs[3], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < innerRows_local; i++)
      d[IDX(i + 1, numCols - 1, stride)] = recvBuf[i];
  }
}
