#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ITER_TIMES 200

// Helper: Access 1D array as 2D
#define IDX(i, j, stride) ((i) * (stride) + (j))

double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col);
void MatrixAdd(double *A, double *B, double a, double b, double *C, int rows,
               int cols, int strideA, int strideB, int strideC, int offAr,
               int offAc, int offBr, int offBc, int offCr, int offCc);
void exchangeBoundaryValues(double *d, int numRows, int numCols, int stride,
                            int *neighborProcs);

int main(int argc, char **argv) {
  /* Initialize MPI */
  int myid, numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (argc < 2) {
    if (myid == 0)
      printf("Usage: %s <n>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  double h = 1.0 / (n + 1);

  // Distribute the mesh points among processes
  int gridDim = (int)sqrt(numprocs);
  if (gridDim * gridDim != numprocs) {
    if (myid == 0)
      printf("Error: Number of processes must be a perfect square.\n");
    MPI_Finalize();
    return 1;
  }

  int row = myid / gridDim; // row index
  int col = myid % gridDim; // column index
  int neighborProcs[4];     // up, down, left, right
  neighborProcs[0] = (myid - gridDim >= 0) ? myid - gridDim : -1;
  neighborProcs[1] = (myid + gridDim < numprocs) ? myid + gridDim : -1;
  neighborProcs[2] = (myid % gridDim != 0) ? myid - 1 : -1;
  neighborProcs[3] = (myid % gridDim != gridDim - 1) ? myid + 1 : -1;

  // Calculate start and end index of the mesh points' block
  int blockSize = n / gridDim;
  int residual = n % gridDim;
  int numRows = blockSize;
  int numCols = blockSize;

  int I_START, J_START;

  if (row < residual) {
    numRows++;
    I_START = row * numRows + 1;
  } else {
    I_START = row * (blockSize + 1) + (residual - (row > residual ? 0 : 0)) +
              1; // Correction
    I_START = row * blockSize + residual + 1;
  }
  // Correct logic for Start Calculation based on original code logic:
  if (row < residual)
    I_START = row * (blockSize + 1) + 1;
  else
    I_START = residual * (blockSize + 1) + (row - residual) * blockSize + 1;

  if (col < residual)
    numCols++;

  if (col < residual)
    J_START = col * (blockSize + 1) + 1;
  else
    J_START = residual * (blockSize + 1) + (col - residual) * blockSize + 1;

  // -------------------------------------------------------------------------
  // Memory Allocation (FLATTENED 1D ARRAYS)
  // -------------------------------------------------------------------------
  // b, u, g, q only need the local "inner" points size: numRows * numCols
  double *b = (double *)malloc(numRows * numCols * sizeof(double));
  double *u = (double *)malloc(numRows * numCols * sizeof(double));
  double *g = (double *)malloc(numRows * numCols * sizeof(double));
  double *q = (double *)malloc(numRows * numCols * sizeof(double));

  // d needs ghost layers: (numRows + 2) * (numCols + 2)
  int extRows = numRows + 2;
  int extCols = numCols + 2;
  double *d =
      (double *)calloc(extRows * extCols, sizeof(double)); // Calloc inits to 0

  if (!b || !u || !g || !q || !d) {
    fprintf(stderr, "Memory allocation failed.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Initialize b
  double x, y;
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      x = (I_START + i) * h;
      y = (J_START + j) * h;
      b[IDX(i, j, numCols)] = 2 * h * h * (x * (1 - x) + y * (1 - y));
    }
  }

  // Start time measurement
  double startTime, endTime;
  startTime = MPI_Wtime();

  // 1.1 u = 0, 1.2 g = -b
  for (int i = 0; i < numRows * numCols; i++) {
    u[i] = 0;
    g[i] = -b[i];
  }

  // 1.3 d = b (inner part of d)
  // Map d[i+1][j+1] = b[i][j]
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      d[IDX(i + 1, j + 1, extCols)] = b[IDX(i, j, numCols)];
    }
  }
  // Boundary of d is already 0 thanks to calloc

  // 1.4 q0 = g^T * g
  double q0 = MatrixDotProduct(g, g, numRows, numCols, numCols, numCols, 0, 0);
  MPI_Allreduce(MPI_IN_PLACE, &q0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // 2. CG iteration
  int iter = 0;
  double tau, q1, beta, dotProduct;
  while (iter < ITER_TIMES) {
    // 2.1 q = Ad
    exchangeBoundaryValues(d, extRows, extCols, extCols, neighborProcs);

    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        // Laplacian stencil on d
        // d[i+1][j], d[i][j+1], d[i+1][j+2], d[i+2][j+1]
        // Center is d[i+1][j+1]
        int center = IDX(i + 1, j + 1, extCols);
        int up = IDX(i, j + 1, extCols);
        int down = IDX(i + 2, j + 1, extCols);
        int left = IDX(i + 1, j, extCols);
        int right = IDX(i + 1, j + 2, extCols);

        q[IDX(i, j, numCols)] =
            -1 * (d[up] + d[left] + d[right] + d[down]) + 4 * d[center];
      }
    }

    // 2.2 tau = q0 / d^T * q
    // Note: d inner part starts at (1,1) in extCols stride
    dotProduct =
        MatrixDotProduct(d, q, numRows, numCols, extCols, numCols, 1, 1);
    MPI_Allreduce(MPI_IN_PLACE, &dotProduct, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    tau = q0 / dotProduct;

    // 2.3 u = u + tau * d
    MatrixAdd(u, d, 1.0, tau, u, numRows, numCols, numCols, extCols, numCols, 0,
              0, 1, 1, 0, 0);

    // 2.4 g = g + tau * q
    MatrixAdd(g, q, 1.0, tau, g, numRows, numCols, numCols, numCols, numCols, 0,
              0, 0, 0, 0, 0);

    // 2.5 q1 = g^T * g
    q1 = MatrixDotProduct(g, g, numRows, numCols, numCols, numCols, 0, 0);
    MPI_Allreduce(MPI_IN_PLACE, &q1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 2.6 beta = q1 / q0;
    beta = q1 / q0;

    // 2.6 d = -g + beta * d
    MatrixAdd(g, d, -1.0, beta, d, numRows, numCols, numCols, extCols, extCols,
              0, 0, 1, 1, 1, 1);

    // 2.7 q0 = q1
    q0 = q1;

    iter++;
  }

  // 3. Calculate the norm of g
  double norm;
  double localDot =
      MatrixDotProduct(g, g, numRows, numCols, numCols, numCols, 0, 0);
  MPI_Reduce(&localDot, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  endTime = MPI_Wtime();
  double exe_time = endTime - startTime;
  double max_exe_time;
  MPI_Reduce(&exe_time, &max_exe_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (myid == 0) {
    printf("The norm of the vector g is %.10e.\n", sqrt(norm));
    printf("Time: %f\n", max_exe_time);
  }

  free(d);
  free(b);
  free(u);
  free(g);
  free(q);

  MPI_Finalize();
  return 0;
}

double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col) {
  double sum = 0.0;
  // Calculate initial pointer offsets
  int startA = IDX(offsetA_row, offsetA_col, strideA);
  int startB = 0; // Assuming B always starts at 0 for now based on usage, or
                  // add offset params for B

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
                            int *neighborProcs) {
  // numRows/numCols here are EXTERNAL dimensions of d

  // Up/Down exchanges are contiguous in memory (rows)
  // Up: Send row 1, Recv into row 0
  if (neighborProcs[0] != -1) {
    MPI_Sendrecv(&d[IDX(1, 0, stride)], numCols, MPI_DOUBLE, neighborProcs[0],
                 0, &d[IDX(0, 0, stride)], numCols, MPI_DOUBLE,
                 neighborProcs[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // Down: Send row numRows-2, Recv into row numRows-1
  if (neighborProcs[1] != -1) {
    MPI_Sendrecv(&d[IDX(numRows - 2, 0, stride)], numCols, MPI_DOUBLE,
                 neighborProcs[1], 0, &d[IDX(numRows - 1, 0, stride)], numCols,
                 MPI_DOUBLE, neighborProcs[1], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }

  // Left/Right exchanges are NON-contiguous (strided)
  // We need to pack them using MPI Derived Datatypes for max performance,
  // but for simplicity/robustness in interview code without IOV quirks,
  // we use a temp buffer (Manual Packing) which is standard practice if
  // avoiding derived types.

  // However, since we refactored to 1D, derived vector types are easy.
  // Let's stick to manual packing for clarity and safety as requested in
  // previous analysis critique (it mentioned copying is overhead, but safety
  // first). Actually, manual packing is better than the previous Loop because
  // now we copy from 1D array.

  int innerRows = numRows - 2; // actual data rows
  double *sendBuf = (double *)malloc(innerRows * sizeof(double));
  double *recvBuf = (double *)malloc(innerRows * sizeof(double));

  // Left: Send column 1, Recv into column 0
  if (neighborProcs[2] != -1) {
    for (int i = 0; i < innerRows; i++)
      sendBuf[i] = d[IDX(i + 1, 1, stride)];

    MPI_Sendrecv(sendBuf, innerRows, MPI_DOUBLE, neighborProcs[2], 0, recvBuf,
                 innerRows, MPI_DOUBLE, neighborProcs[2], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (int i = 0; i < innerRows; i++)
      d[IDX(i + 1, 0, stride)] = recvBuf[i];
  }

  // Right: Send column numCols-2, Recv into column numCols-1
  if (neighborProcs[3] != -1) {
    for (int i = 0; i < innerRows; i++)
      sendBuf[i] = d[IDX(i + 1, numCols - 2, stride)];

    MPI_Sendrecv(sendBuf, innerRows, MPI_DOUBLE, neighborProcs[3], 0, recvBuf,
                 innerRows, MPI_DOUBLE, neighborProcs[3], 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (int i = 0; i < innerRows; i++)
      d[IDX(i + 1, numCols - 1, stride)] = recvBuf[i];
  }

  free(sendBuf);
  free(recvBuf);
}