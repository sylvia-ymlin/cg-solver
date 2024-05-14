/*
 * Parallel PCG solver for the 2D Poisson equation on the unit square.
 *
 * Discretizes -∆u = f with homogeneous Dirichlet boundary conditions using
 * a 5-point finite difference stencil on an n×n interior grid (h = 1/(n+1)).
 * The global grid is distributed across a sqrt(p)×sqrt(p) process grid; each
 * process owns a subgrid and exchanges halo values with its four neighbors
 * before every stencil application. The linear system Au = h²f is solved with
 * Preconditioned Conjugate Gradient (PCG) using a Jacobi preconditioner.
 *
 * Usage: mpirun -n <p> ./solver <n> [max_iter] [tol] [jacobi_iters] [problem]
 *   problem: 0 = quadratic  u = x(1-x)y(1-y)  (default)
 *            1 = sinusoidal  u = sin(πx)sin(πy)
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_ITER 200
#define DEFAULT_TOL 1e-6

double MatrixDotProduct(double** A, double** B, int rows, int cols);
void MatrixAdd(double** A, double** B, double a, double b, double** C, int rows, int cols);
void exchangeBoundaryValues(double** d, int numRows, int numCols, int* neighborProcs);
void ApplyStencil(double** q, double** p, int numRows, int numCols);
void ApplyJacobiPrecond(double** z, double** r, int numRows, int numCols, int iters);


int main(int argc, char** argv) {
    /* Initialize MPI */
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Parse arguments */
    if (argc >= 2 && strcmp(argv[1], "--help") == 0) {
        if (myid == 0) {
            printf("Usage: %s <n> [max_iter] [tol] [jacobi_iters] [problem]\n"
                   "  problem: 0 = quadratic (default), 1 = sinusoidal\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (argc < 2) {
        if (myid == 0) {
            printf("Usage: %s <n> [max_iter] [tol] [jacobi_iters] [problem]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int max_iter = (argc >= 3) ? atoi(argv[2]) : DEFAULT_MAX_ITER;
    double tol = (argc >= 4) ? atof(argv[3]) : DEFAULT_TOL;
    int jacobi_iters = (argc >= 5) ? atoi(argv[4]) : 5;
    int problem = (argc >= 6) ? atoi(argv[5]) : 0;
    double h = 1.0 / (n + 1);
    double pi = acos(-1.0);

    /* Set up the 2D process grid and determine this process's subgrid */
    int gridDim = (int)sqrt(numprocs);  // Assume numprocs is a perfect square
    if (gridDim * gridDim != numprocs) {
        if (myid == 0) {
            printf("Error: Number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int row = myid / gridDim;  // row index in process grid
    int col = myid % gridDim;  // column index in process grid
    int neighborProcs[4];      // up, down, left, right
    neighborProcs[0] = (row > 0) ? myid - gridDim : -1;
    neighborProcs[1] = (row < gridDim - 1) ? myid + gridDim : -1;
    neighborProcs[2] = (col > 0) ? myid - 1 : -1;
    neighborProcs[3] = (col < gridDim - 1) ? myid + 1 : -1;

    // Calculate start index and size of each process's block
    int blockSize = n / gridDim;
    int residual = n % gridDim;
    int numRows = blockSize, numCols = blockSize;
    int I_START, J_START;

    if (row < residual) {
        numRows++;
        I_START = row * (blockSize + 1) + 1;
    } else {
        I_START = residual * (blockSize + 1) + (row - residual) * blockSize + 1;
    }

    if (col < residual) {
        numCols++;
        J_START = col * (blockSize + 1) + 1;
    } else {
        J_START = residual * (blockSize + 1) + (col - residual) * blockSize + 1;
    }

    /* Allocate all vectors as extended (numRows+2) x (numCols+2) matrices.
     * Interior elements are accessed as [i+1][j+1]; the outer ring of cells
     * serves as ghost cells for halo exchange. */
    int extRows = numRows + 2;
    int extCols = numCols + 2;
    double** b = (double**)malloc(extRows * sizeof(double*));
    double** x = (double**)malloc(extRows * sizeof(double*));
    double** r = (double**)malloc(extRows * sizeof(double*));
    double** z = (double**)malloc(extRows * sizeof(double*));
    double** p = (double**)malloc(extRows * sizeof(double*));
    double** q = (double**)malloc(extRows * sizeof(double*));
    for (int i = 0; i < extRows; i++) {
        b[i] = (double*)calloc(extCols, sizeof(double));
        x[i] = (double*)calloc(extCols, sizeof(double));
        r[i] = (double*)calloc(extCols, sizeof(double));
        z[i] = (double*)calloc(extCols, sizeof(double));
        p[i] = (double*)calloc(extCols, sizeof(double));
        q[i] = (double*)calloc(extCols, sizeof(double));
    }

    /* Initialize the right-hand side b = h²f at each interior mesh point */
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            double xx = (I_START + i) * h;
            double yy = (J_START + j) * h;
            if (problem == 0) {
                b[i + 1][j + 1] = 2 * h * h * (xx * (1 - xx) + yy * (1 - yy));
            } else {
                b[i + 1][j + 1] = 2 * pi * pi * h * h * sin(pi * xx) * sin(pi * yy);
            }
        }
    }

    // Start time measurement
    double startTime = MPI_Wtime();

    // 1. PCG initialization
    // 1.1 x = 0 (already zero from calloc)
    // 1.2 r = b
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            r[i + 1][j + 1] = b[i + 1][j + 1];
        }
    }

    // 1.3 z = M^{-1} r
    ApplyJacobiPrecond(z, r, numRows, numCols, jacobi_iters);

    // 1.4 p = z
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            p[i + 1][j + 1] = z[i + 1][j + 1];
        }
    }

    // 1.5 rho = r^T * z
    double rho = MatrixDotProduct(r, z, numRows, numCols);
    MPI_Allreduce(MPI_IN_PLACE, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 2. PCG iteration
    int iter = 0;
    double alpha, rho_new, beta, dotProduct;
    while (iter < max_iter) {
        // 2.1 q = Ap
        exchangeBoundaryValues(p, extRows, extCols, neighborProcs);
        ApplyStencil(q, p, numRows, numCols);

        // 2.2 alpha = rho / p^T * q
        dotProduct = MatrixDotProduct(p, q, numRows, numCols);
        MPI_Allreduce(MPI_IN_PLACE, &dotProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = rho / dotProduct;

        // 2.3 x = x + alpha * p
        MatrixAdd(x, p, 1.0, alpha, x, numRows, numCols);

        // 2.4 r = r - alpha * q
        MatrixAdd(r, q, 1.0, -alpha, r, numRows, numCols);

        // 2.5 Check convergence
        dotProduct = MatrixDotProduct(r, r, numRows, numCols);
        MPI_Allreduce(MPI_IN_PLACE, &dotProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (tol > 0.0 && sqrt(dotProduct) < tol) {
            iter++;
            break;
        }

        // 2.6 z = M^{-1} r
        ApplyJacobiPrecond(z, r, numRows, numCols, jacobi_iters);

        // 2.7 rho_new = r^T * z
        rho_new = MatrixDotProduct(r, z, numRows, numCols);
        MPI_Allreduce(MPI_IN_PLACE, &rho_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 2.8 beta = rho_new / rho
        beta = rho_new / rho;

        // 2.9 p = z + beta * p
        MatrixAdd(z, p, 1.0, beta, p, numRows, numCols);

        // 2.10 rho = rho_new
        rho = rho_new;

        iter++;
    }

    if (myid == 0) {
        printf("Iterations: %d\n", iter);
        printf("Total Time: %f\n", MPI_Wtime() - startTime);
    }

    // 3. Compute discrete L2 error against the exact solution
    double local_sq_err = 0.0, global_sq_err = 0.0;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            double xx = (I_START + i) * h;
            double yy = (J_START + j) * h;
            double u_exact, err;
            if (problem == 0) {
                u_exact = xx * (1 - xx) * yy * (1 - yy);
            } else {
                u_exact = sin(pi * xx) * sin(pi * yy);
            }
            err = u_exact - x[i + 1][j + 1];
            local_sq_err += err * err;
        }
    }
    MPI_Reduce(&local_sq_err, &global_sq_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        printf("L2 Error: %e\n", h * sqrt(global_sq_err));
    }

    // Free memory
    for (int i = 0; i < extRows; i++) {
        free(b[i]); free(x[i]); free(r[i]);
        free(z[i]); free(p[i]); free(q[i]);
    }
    free(b); free(x); free(r);
    free(z); free(p); free(q);

    MPI_Finalize();
    return 0;
}

// Compute the local dot product of interior elements: sum of A[i][j] * B[i][j].
// The caller is responsible for the MPI_Allreduce to get the global sum.
double MatrixDotProduct(double** A, double** B, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += A[i + 1][j + 1] * B[i + 1][j + 1];
        }
    }
    return sum;
}

// Compute C = a*A + b*B on interior elements. Safe for in-place use (C may
// alias A or B) because each element is read before it is overwritten.
void MatrixAdd(double** A, double** B, double a, double b, double** C, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i + 1][j + 1] = a * A[i + 1][j + 1] + b * B[i + 1][j + 1];
        }
    }
}

// Exchange interior boundary rows/columns with neighboring processes.
// numRows and numCols include the 2 ghost layers (i.e. extRows, extCols).
void exchangeBoundaryValues(double** d, int numRows, int numCols, int* neighborProcs) {
    int innerRows = numRows - 2;
    int innerCols = numCols - 2;

    if (neighborProcs[0] != -1) {  // up
        MPI_Sendrecv(&d[1][1], innerCols, MPI_DOUBLE, neighborProcs[0], 0,
                     &d[0][1], innerCols, MPI_DOUBLE, neighborProcs[0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighborProcs[1] != -1) {  // down
        MPI_Sendrecv(&d[numRows - 2][1], innerCols, MPI_DOUBLE, neighborProcs[1], 0,
                     &d[numRows - 1][1], innerCols, MPI_DOUBLE, neighborProcs[1], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighborProcs[2] != -1) {  // left
        double leftSend[innerRows], leftRecv[innerRows];
        for (int i = 0; i < innerRows; i++) {
            leftSend[i] = d[i + 1][1];
        }
        MPI_Sendrecv(leftSend, innerRows, MPI_DOUBLE, neighborProcs[2], 0,
                     leftRecv, innerRows, MPI_DOUBLE, neighborProcs[2], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < innerRows; i++) {
            d[i + 1][0] = leftRecv[i];
        }
    }
    if (neighborProcs[3] != -1) {  // right
        double rightSend[innerRows], rightRecv[innerRows];
        for (int i = 0; i < innerRows; i++) {
            rightSend[i] = d[i + 1][numCols - 2];
        }
        MPI_Sendrecv(rightSend, innerRows, MPI_DOUBLE, neighborProcs[3], 0,
                     rightRecv, innerRows, MPI_DOUBLE, neighborProcs[3], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < innerRows; i++) {
            d[i + 1][numCols - 1] = rightRecv[i];
        }
    }
}

// Apply the matrix-free 5-point Laplacian stencil: q = Ap.
// Reads ghost cells of p (must be filled by exchangeBoundaryValues first).
void ApplyStencil(double** q, double** p, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            q[i + 1][j + 1] = 4 * p[i + 1][j + 1]
                             - (p[i][j + 1] + p[i + 2][j + 1]
                             +  p[i + 1][j] + p[i + 1][j + 2]);
        }
    }
}

// Apply the Jacobi preconditioner: compute z ≈ A^{-1} r using iters iterations.
// Ghost cells of z remain zero (local approximation, no MPI exchange).
void ApplyJacobiPrecond(double** z, double** r, int numRows, int numCols, int iters) {
    int extRows = numRows + 2;
    int extCols = numCols + 2;
    int i, j, k;
    double** tmp;

    // Allocate a scratch buffer for the Jacobi iteration
    double** buf = (double**)malloc(extRows * sizeof(double*));
    for (i = 0; i < extRows; i++) {
        buf[i] = (double*)calloc(extCols, sizeof(double));
    }

    // Initialize z = D^{-1} r, where D is the stencil diagonal (4.0)
    for (i = 0; i < extRows; i++) {
        for (j = 0; j < extCols; j++) {
            z[i][j] = r[i][j] / 4.0;
        }
    }

    // Alternately write into buf and z; cur always holds the latest result
    double** cur = z;
    double** nxt = buf;
    for (k = 0; k < iters; k++) {
        for (i = 0; i < numRows; i++) {
            for (j = 0; j < numCols; j++) {
                nxt[i + 1][j + 1] = (r[i + 1][j + 1]
                                   + cur[i][j + 1] + cur[i + 2][j + 1]
                                   + cur[i + 1][j] + cur[i + 1][j + 2]) / 4.0;
            }
        }
        tmp = cur; cur = nxt; nxt = tmp;
    }

    // If the result is in buf (odd iters), copy it back into z
    if (cur != z) {
        for (i = 0; i < extRows; i++) {
            for (j = 0; j < extCols; j++) {
                z[i][j] = cur[i][j];
            }
        }
    }

    for (i = 0; i < extRows; i++) {
        free(buf[i]);
    }
    free(buf);
}
