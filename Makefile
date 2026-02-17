# Compiler settings
CC = mpicc
CFLAGS = -std=c99 -g -O3 -Wall -Wextra -Iinclude
LIBS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source search path
VPATH = $(SRC_DIR)

# Targets (placed in build dir)
BINS = $(BUILD_DIR)/CG $(BUILD_DIR)/PCG
PROF_BIN = $(BUILD_DIR)/cg_prof

# Object files (placed in build dir)
UTILS_OBJS = $(BUILD_DIR)/solver_utils.o
PROF_OBJS = $(BUILD_DIR)/cg_solver.o $(BUILD_DIR)/linalg.o $(BUILD_DIR)/comm.o $(BUILD_DIR)/precond.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/solver_utils.o
CG_OBJS = $(BUILD_DIR)/CG_baseline.o $(UTILS_OBJS)
PCG_OBJS = $(BUILD_DIR)/PCG.o $(UTILS_OBJS)

all: $(BINS) $(PROF_BIN)

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Generic rule for object files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Profiling Sandbox
$(PROF_BIN): $(PROF_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Legacy Baseline
$(BUILD_DIR)/CG: $(CG_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Legacy PCG
$(BUILD_DIR)/PCG: $(PCG_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -rf *.dSYM
	$(RM) CG PCG cg_prof *.o # Clean up old files in root if they exist
