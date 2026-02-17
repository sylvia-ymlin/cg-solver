// here we define some helper functions for the solver
#ifndef SOLVER_UTILS_H
#define SOLVER_UTILS_H

#include <mpi.h>

// default parameters, max iterations and tolerance used in the solver to check
// for convergence
#define DEFAULT_MAX_ITER 200
#define DEFAULT_TOL 1e-6

// Helper: Access 1D array as 2D
#define IDX(i, j, stride) ((i) * (stride) + (j))

// Grid context structure, used to store the grid information, avoid passing
// parameters one by one
typedef struct {
  int myid;     // process id (from MPI_Comm_rank)
  int numprocs; // number of processes (from MPI_Comm_size)
  int n;        // global grid size (user input)
  int gridDim;  // grid dimension (= sqrt(numprocs), for 2D decomposition)
  int row, col; // row and column of this process in the 2D grid
                // (row = myid / gridDim, col = myid % gridDim)
  int neighborProcs[4]; // neighbor process IDs: [up, down, left, right]
                        // -1 if no neighbor in that direction (boundary)
  int numRows, numCols; // local grid size for this process (varies with load)
  int I_START, J_START; // global start indices for this process's subdomain
  double h;             // grid spacing (= 1.0 / (n + 1))
} GridContext;

// three helper functions

// setup the grid context
int SetupGrid(int n, GridContext *ctx);

// matrix dot product
double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col);

// exchange boundary values
void exchangeBoundaryValues(double *d, int numRows, int numCols, int stride,
                            int *neighborProcs, double *sendBuf,
                            double *recvBuf);

#endif // SOLVER_UTILS_H
