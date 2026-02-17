#ifndef COMM_H
#define COMM_H

#include "solver_structs.h"

// Performs MPI_Sendrecv to exchange boundary values
// Profiles communication latency/bandwidth
void halo_exchange(Vector *v, SolverContext *ctx);

// Calculates global dot product using MPI_Allreduce
// Profiles collective communication
double dot_product_global(Vector *a, Vector *b, SolverContext *ctx);

#endif // COMM_H
