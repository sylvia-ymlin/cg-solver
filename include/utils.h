#ifndef UTILS_H
#define UTILS_H

#include "solver_structs.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// profiling macros
// TIC: Start timing updates start_time
// TOC: Stop timing, adds elapsed difference to accumulator
#define TIC(start_var) (*(start_var) = MPI_Wtime())
#define TOC(start_var, accum_var) (*(accum_var) += (MPI_Wtime() - *(start_var)))

// Memory Management
Vector *create_vector(int num_rows, int num_cols);
void free_vector(Vector *vec);
void copy_vector(Vector *dst, Vector *src);

// Profiling Report
void print_profiling_results(SolverContext *ctx);

#endif // UTILS_H
