# Compiler settings
CC = mpicc
CFLAGS = -std=c99 -g -O3 -Wall -Wextra
LIBS = -lm

# Directories
BUILD_DIR = build

# Targets
SOLVER_BIN = solver

SOLVER_OBJS = $(BUILD_DIR)/main.o

all: $(SOLVER_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SOLVER_BIN): $(SOLVER_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -rf *.dSYM
	$(RM) $(SOLVER_BIN)

check: all
	mpirun -n 1 ./solver 64 200 1e-6 5 0
	mpirun -n 4 ./solver 64 200 1e-6 5 0

.PHONY: all clean check
