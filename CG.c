#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_MAX_ITER 200
#define DEFAULT_TOL 1e-6

// Helper: Access 1D array as 2D
#define IDX(i, j, stride) ((i) * (stride) + (j))

double MatrixDotProduct(double *A, double *B, int rows, int cols, int strideA,
                        int strideB, int offsetA_row, int offsetA_col);
void MatrixAdd(double *A, double *B, double a, double b, double *C, int rows,
               int cols, int strideA, int strideB, int strideC, int offAr,
               int offAc, int offBr, int offBc, int offCr, int offCc);
void exchangeBoundaryValues(double *d, int numRows, int numCols, int stride,
                            int *neighborProcs,
                            double *sendBuf, double *recvBuf);

int main(int argc, char **argv) {
  /* Initialize MPI */
  int myid, numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (argc < 2) {
    if (myid == 0)
      printf("Usage: %s <n> [max_iter] [tol]\n"
             "  n        : grid intervals (required)\n"
             "  max_iter : max CG iterations (default %d, 0 = use tol only)\n"
             "  tol      : residual tolerance (default %.1e, 0 = use max_iter only)\n",
             argv[0], DEFAULT_MAX_ITER, DEFAULT_TOL);
    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  if (n <= 0) {
    if (myid == 0) printf("Error: n must be positive.\n");
    MPI_Finalize();
    return 1;
  }
  int max_iter = (argc >= 3) ? atoi(argv[2]) : DEFAULT_MAX_ITER;
  double tol = (argc >= 4) ? atof(argv[3]) : 0.0; // default: no convergence check (timing mode)
  double h = 1.0 / (n + 1);

  // Distribute the mesh points among processes
  int gridDim = (int)sqrt(numprocs);
  if (gridDim * gridDim != numprocs) {
    if (myid == 0)
      printf("Error: Number of processes must be a perfect square.\n");
    MPI_Finalize();
    return 1;
  }

  if (n < gridDim) {
    if (myid == 0)
      printf("Error: n (%d) must be >= sqrt(numprocs) (%d).\n", n, gridDim);
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
    I_START = row * (blockSize + 1) + 1;
  } else {
    I_START = residual * (blockSize + 1) + (row - residual) * blockSize + 1;
  }

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

  // Pre-allocate halo exchange buffers (avoid malloc inside iteration loop)
  int innerRows = extRows - 2;
  double *sendBuf = (double *)malloc(innerRows * sizeof(double));
  double *recvBuf = (double *)malloc(innerRows * sizeof(double));
  if (!sendBuf || !recvBuf) {
    fprintf(stderr, "Halo buffer allocation failed.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Start time measurement
  double startTime, endTime;
  double t_halo = 0.0, t_reduce = 0.0, t_comp = 0.0, t0; // timing breakdown
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
  int converged = 0;
  double tau, q1, beta, dotProduct;
  while (iter < max_iter) {
    // 2.1 q = Ad (halo exchange + stencil)
    t0 = MPI_Wtime();
    exchangeBoundaryValues(d, extRows, extCols, extCols, neighborProcs,
                           sendBuf, recvBuf);
    t_halo += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        // 5-point Laplacian stencil: q(i,j) = 4*d(i,j) - d(i-1,j) - d(i+1,j)
        //                                                - d(i,j-1) - d(i,j+1)
        //
        // Index shift: d has ghost layers, so the interior point (i,j)
        // in the local grid maps to d[i+1][j+1] in the extended array.
        // This means:
        //   center = d[i+1, j+1]   (the point itself)
        //   up     = d[i,   j+1]   (row above in extended coords)
        //   down   = d[i+2, j+1]   (row below)
        //   left   = d[i+1, j  ]   (column left)
        //   right  = d[i+1, j+2]   (column right)
        int center = IDX(i + 1, j + 1, extCols);
        int up = IDX(i, j + 1, extCols);
        int down = IDX(i + 2, j + 1, extCols);
        int left = IDX(i + 1, j, extCols);
        int right = IDX(i + 1, j + 2, extCols);

        q[IDX(i, j, numCols)] =
            -1 * (d[up] + d[left] + d[right] + d[down]) + 4 * d[center];
      }
    }

    // 2.2 tau = q0 / d^T * q (local dot product + global reduction)
    dotProduct =
        MatrixDotProduct(d, q, numRows, numCols, extCols, numCols, 1, 1);
    t_comp += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &dotProduct, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    t_reduce += MPI_Wtime() - t0;
    tau = q0 / dotProduct;

    // 2.3 u = u + tau * d, 2.4 g = g + tau * q (local vector ops)
    t0 = MPI_Wtime();
    MatrixAdd(u, d, 1.0, tau, u, numRows, numCols, numCols, extCols, numCols, 0,
              0, 1, 1, 0, 0);
    MatrixAdd(g, q, 1.0, tau, g, numRows, numCols, numCols, numCols, numCols, 0,
              0, 0, 0, 0, 0);

    // 2.5 q1 = g^T * g (local dot product)
    q1 = MatrixDotProduct(g, g, numRows, numCols, numCols, numCols, 0, 0);
    t_comp += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &q1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t_reduce += MPI_Wtime() - t0;

    // Check convergence: ||g||_2 = sqrt(q1) < tol
    if (tol > 0.0 && sqrt(q1) < tol) {
      converged = 1;
      iter++;
      break;
    }

    // 2.6 beta = q1 / q0;
    beta = q1 / q0;

    // 2.6 d = -g + beta * d
    MatrixAdd(g, d, -1.0, beta, d, numRows, numCols, numCols, extCols, extCols,
              0, 0, 1, 1, 1, 1);

    // 2.7 q0 = q1
    q0 = q1;

    iter++;
  }

  // 3. Calculate residual norm
  double norm;
  double localDot =
      MatrixDotProduct(g, g, numRows, numCols, numCols, numCols, 0, 0);
  MPI_Reduce(&localDot, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // 4. Compute L2 error vs exact solution: u_exact(x,y) = x(1-x)*y(1-y)
  //    Note: CG solves A*u = b where b is scaled by h^2, so u stores the
  //    solution in the same scaling. The exact u_exact also gets h^2 scaling
  //    from the RHS construction, so we compare directly.
  //    Actually, the CG solves the system where the matrix is the raw stencil
  //    (coefficients 4,-1,-1,-1,-1) and b = 2*h^2*(x(1-x)+y(1-y)).
  //    The exact solution to this system is u_exact(i,j) = h^2 * x(1-x)*y(1-y),
  //    but since b already has h^2 baked in, we need to verify numerically.
  //    Simpler: just compare u against the expected discrete solution.
  double local_l2err = 0.0;
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      x = (I_START + i) * h;
      y = (J_START + j) * h;
      double u_exact = x * (1.0 - x) * y * (1.0 - y);
      double diff = u[IDX(i, j, numCols)] - u_exact;
      local_l2err += diff * diff;
    }
  }
  double global_l2err;
  MPI_Reduce(&local_l2err, &global_l2err, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  endTime = MPI_Wtime();
  double exe_time = endTime - startTime;
  double max_exe_time;
  MPI_Reduce(&exe_time, &max_exe_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  // Gather timing breakdown (max across processes)
  double max_t_halo, max_t_reduce, max_t_comp;
  MPI_Reduce(&t_halo, &max_t_halo, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_reduce, &max_t_reduce, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&t_comp, &max_t_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    printf("Iterations: %d\n", iter);
    printf("Converged: %s\n", converged ? "yes" : "no");
    printf("Residual: %.10e\n", sqrt(norm));
    printf("L2_Error: %.10e\n", sqrt(global_l2err) * h); // L2 norm scaled by h
    printf("Time: %f\n", max_exe_time);
    printf("Time_Halo: %f\n", max_t_halo);
    printf("Time_Reduce: %f\n", max_t_reduce);
    printf("Time_Comp: %f\n", max_t_comp);
  }

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
                            int *neighborProcs,
                            double *sendBuf, double *recvBuf) {
  // numRows/numCols here are EXTERNAL dimensions of d
  // sendBuf/recvBuf are pre-allocated by caller (size >= numRows-2)

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

  // Left/Right: columns are non-contiguous, so we manually pack into a buffer.
  // Could use MPI derived types instead, but this is simpler.
  int innerRows_local = numRows - 2;

  // Left: Send column 1, Recv into column 0
  if (neighborProcs[2] != -1) {
    for (int i = 0; i < innerRows_local; i++)
      sendBuf[i] = d[IDX(i + 1, 1, stride)];

    MPI_Sendrecv(sendBuf, innerRows_local, MPI_DOUBLE, neighborProcs[2], 0,
                 recvBuf, innerRows_local, MPI_DOUBLE, neighborProcs[2], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < innerRows_local; i++)
      d[IDX(i + 1, 0, stride)] = recvBuf[i];
  }

  // Right: Send column numCols-2, Recv into column numCols-1
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