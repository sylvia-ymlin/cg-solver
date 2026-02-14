###############################################################################
# Makefile: Parallel CG method
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -g -O3 -Wall -Wextra
LIBS = -lm

BINS = CG PCG
UTILS = solver_utils.o

all: $(BINS)

%.o: %.c solver_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

CG: CG_baseline.c $(UTILS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

PCG: PCG.c $(UTILS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(BINS) $(UTILS)
