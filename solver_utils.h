#ifndef SOLVER_UTILS_H
#define SOLVER_UTILS_H

#include <mpi.h>

#define DEFAULT_MAX_ITER 200
#define DEFAULT_TOL 1e-6

// Helper: Access 1D array as 2D
#define IDX(i, j, stride) ((i) * (stride) + (j))

typedef struct {
  int myid;
  int numprocs;
  int n;
  int gridDim;
  int row, col;
  int neighborProcs[4];
  int numRows, numCols;
  int I_START, J_START;
  double h;
} GridContext;

int SetupGrid(int n, GridContext *ctx);

double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col);

void MatrixAdd(double *A, double *B, double a, double b, double *C, int rows,
               int cols, int strideA, int strideB, int strideC, int offAr,
               int offAc, int offBr, int offBc, int offCr, int offCc);

void exchangeBoundaryValues(double *d, int numRows, int numCols, int stride,
                            int *neighborProcs, double *sendBuf,
                            double *recvBuf);

#endif // SOLVER_UTILS_H
