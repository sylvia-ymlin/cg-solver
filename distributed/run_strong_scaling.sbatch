#!/bin/bash
#===============================================================================
# Strong Scaling Experiment for MPI CG Solver
#
# Purpose: Measure how solution time scales with fixed problem size
#          as the number of processes increases.
#
# Key metrics:
#   - Speedup: S(p) = T(1) / T(p)
#   - Efficiency: E(p) = S(p) / p
#   - Communication/Compute ratio across process counts
#===============================================================================

#SBATCH -A uppmax2024-2-9
#SBATCH -J cg_strong
#SBATCH -t 01:00:00
#SBATCH --nodes=1-16
#SBATCH --ntasks-per-node=20
#SBATCH -o results/strong_scaling_%j.out
#SBATCH -e results/strong_scaling_%j.err

# Load modules
module load gcc/12.2.0
module load openmpi/4.1.4

# Configuration
PROBLEM_SIZE=4096        # Fixed global grid size
MAX_ITER=100             # Iterations for timing (not convergence)
REPEAT=3                 # Number of runs for averaging

cd $SLURM_SUBMIT_DIR

# Ensure results directory exists
mkdir -p results

# Output file
RESULT_FILE="results/strong_scaling_$(date +%Y%m%d_%H%M%S).tsv"
echo "Strong Scaling Experiment - $(date)" > $RESULT_FILE
echo "Problem Size: ${PROBLEM_SIZE}x${PROBLEM_SIZE}" >> $RESULT_FILE
echo -e "np\tnodes\ttime_s\titer_time_ms\tstencil_ms\thalo_ms\tallreduce_ms\taxpy_ms\tspeedup\tefficiency" >> $RESULT_FILE

# Reference time (single process)
echo "=========================================="
echo "Running baseline (np=1)..."
echo "=========================================="

BASELINE_TIME=0
for r in $(seq 1 $REPEAT); do
    OUTPUT=$(mpirun -np 1 ./build/cg_prof $PROBLEM_SIZE $MAX_ITER 2>&1)
    TIME=$(echo "$OUTPUT" | grep -oP 'Total:\s*\K[\d.]+' || echo "0")
    BASELINE_TIME=$(echo "$BASELINE_TIME + $TIME" | bc)
done
BASELINE_TIME=$(echo "scale=6; $BASELINE_TIME / $REPEAT" | bc)
echo "Baseline time (np=1): ${BASELINE_TIME}s"

# Test configurations: process counts that are perfect squares (for 2D decomposition)
# Note: Must be perfect squares for 2D domain decomposition
PROCESS_COUNTS="1 4 9 16 25 36 49 64 81 100 121 144 169 196 225 256"

for NP in $PROCESS_COUNTS; do
    echo ""
    echo "=========================================="
    echo "Running with np=$NP"
    echo "=========================================="
    
    # Calculate nodes needed (assuming 20 cores per node)
    NODES=$(( (NP + 19) / 20 ))
    
    # Skip if requesting more resources than allocated
    if [ $NODES -gt $SLURM_JOB_NUM_NODES ]; then
        echo "Skipping np=$NP (requires $NODES nodes, have $SLURM_JOB_NUM_NODES)"
        continue
    fi
    
    # Run multiple times and average
    TOTAL_TIME=0
    STENCIL_TIME=0
    HALO_TIME=0
    ALLREDUCE_TIME=0
    AXPY_TIME=0
    
    for r in $(seq 1 $REPEAT); do
        OUTPUT=$(mpirun -np $NP ./build/cg_prof $PROBLEM_SIZE $MAX_ITER 2>&1)
        
        # Parse timing from output
        T=$(echo "$OUTPUT" | grep -oP 'Total:\s*\K[\d.]+' | head -1 || echo "0")
        TS=$(echo "$OUTPUT" | grep -oP 'Stencil:\s*\K[\d.]+' | head -1 || echo "0")
        TH=$(echo "$OUTPUT" | grep -oP 'Halo:\s*\K[\d.]+' | head -1 || echo "0")
        TA=$(echo "$OUTPUT" | grep -oP 'Allreduce:\s*\K[\d.]+' | head -1 || echo "0")
        TX=$(echo "$OUTPUT" | grep -oP 'AXPY:\s*\K[\d.]+' | head -1 || echo "0")
        
        TOTAL_TIME=$(echo "$TOTAL_TIME + $T" | bc)
        STENCIL_TIME=$(echo "$STENCIL_TIME + $TS" | bc)
        HALO_TIME=$(echo "$HALO_TIME + $TH" | bc)
        ALLREDUCE_TIME=$(echo "$ALLREDUCE_TIME + $TA" | bc)
        AXPY_TIME=$(echo "$AXPY_TIME + $TX" | bc)
    done
    
    AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $REPEAT" | bc)
    AVG_STENCIL=$(echo "scale=6; $STENCIL_TIME / $REPEAT" | bc)
    AVG_HALO=$(echo "scale=6; $HALO_TIME / $REPEAT" | bc)
    AVG_ALLREDUCE=$(echo "scale=6; $ALLREDUCE_TIME / $REPEAT" | bc)
    AVG_AXPY=$(echo "scale=6; $AXPY_TIME / $REPEAT" | bc)
    
    # Calculate metrics
    ITER_TIME=$(echo "scale=6; $AVG_TIME / $MAX_ITER * 1000" | bc)
    SPEEDUP=$(echo "scale=3; $BASELINE_TIME / $AVG_TIME" | bc)
    EFFICIENCY=$(echo "scale=3; $SPEEDUP / $NP" | bc)
    
    echo "Time: ${AVG_TIME}s, Speedup: ${SPEEDUP}x, Efficiency: ${EFFICIENCY}"
    
    # Save results
    echo -e "${NP}\t${NODES}\t${AVG_TIME}\t${ITER_TIME}\t${AVG_STENCIL}\t${AVG_HALO}\t${AVG_ALLREDUCE}\t${AVG_AXPY}\t${SPEEDUP}\t${EFFICIENCY}" >> $RESULT_FILE
done

echo ""
echo "=========================================="
echo "Strong Scaling Experiment Complete"
echo "Results saved to: $RESULT_FILE"
echo "=========================================="
