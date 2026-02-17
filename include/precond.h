#ifndef PRECOND_H
#define PRECOND_H

#include "solver_structs.h"

// Apply M^{-1} to r, storing result in z
// - PRECOND_NONE: z = r (Copy)
// - PRECOND_JACOBI: z = Jacobi(r)
// Profiles compute/memory intensity
void apply_preconditioner(Vector *z, Vector *r, SolverContext *ctx);

#endif // PRECOND_H
