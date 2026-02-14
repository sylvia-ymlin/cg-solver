###############################################################################
# Makefile: Parallel CG method
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -g -O3 -Wall -Wextra
LIBS = -lm

BIN = CG

all: $(BIN)

$(BIN): CG.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) $(BIN)
