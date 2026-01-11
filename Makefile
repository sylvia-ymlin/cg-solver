###############################################################################
# Makefile Parellel CG method
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -g -O3 -lm
LIBS = -lm

BIN = CG

all: $(BIN)

stencil: CG.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
	
clean:
	$(RM) $(BIN)
