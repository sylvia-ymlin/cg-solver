#ifndef LINALG_H
#define LINALG_H

#include "solver_structs.h"

// y = y + alpha * x
// Profiles memory bandwidth intensity
void vec_axpy(Vector *y, Vector *x, double alpha, SolverContext *ctx);

// y = x + beta * y
// Profiles memory bandwidth intensity (used in search direction update)
void vec_xpay(Vector *y, Vector *x, double beta, SolverContext *ctx);

// Apply 5-point stencil: q = A * p
// Profiles compute intensity (and some Halo access)
void apply_stencil(Vector *q, Vector *p, SolverContext *ctx);

// Set vector to constant value
void vec_set(Vector *v, double val);

// Calculate local dot product (without MPI reduction)
double vec_dot_local(Vector *a, Vector *b);

#endif // LINALG_H
