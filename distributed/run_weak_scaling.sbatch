#!/bin/bash
#===============================================================================
# Weak Scaling Experiment for MPI CG Solver
#
# Purpose: Measure how solution time scales when problem size per process
#          remains constant while the number of processes increases.
#
# Ideal weak scaling: Time remains constant (perfect scalability)
# Real-world: Time increases due to communication overhead
#
# This experiment helps identify the "computation threshold" needed to
# hide communication latency - the minimum local problem size where
# T_compute > K * T_comm.
#===============================================================================

#SBATCH -A uppmax2024-2-9
#SBATCH -J cg_weak
#SBATCH -t 01:00:00
#SBATCH --nodes=1-16
#SBATCH --ntasks-per-node=20
#SBATCH -o results/weak_scaling_%j.out
#SBATCH -e results/weak_scaling_%j.err

# Load modules
module load gcc/12.2.0
module load openmpi/4.1.4

cd $SLURM_SUBMIT_DIR
mkdir -p results

# Configuration
LOCAL_SIZES="128 256 512"   # Local grid size per process (n_local x n_local)
MAX_ITER=200
REPEAT=3

# Output file
RESULT_FILE="results/weak_scaling_$(date +%Y%m%d_%H%M%S).tsv"
echo "Weak Scaling Experiment - $(date)" > $RESULT_FILE
echo -e "local_n\tglobal_n\tnp\tnodes\ttime_s\titer_time_ms\tstencil_ms\thalo_ms\tallreduce_ms\toverhead_pct" >> $RESULT_FILE

echo "=========================================="
echo "Weak Scaling Experiment"
echo "=========================================="

for LOCAL_N in $LOCAL_SIZES; do
    echo ""
    echo "--- Local grid size: ${LOCAL_N}x${LOCAL_N} ---"
    
    # Process counts (perfect squares for 2D decomposition)
    PROCESS_COUNTS="1 4 9 16 25 36 49 64 81 100"
    
    # Get baseline time for np=1
    GLOBAL_N=$LOCAL_N
    echo "Getting baseline (np=1, global=${GLOBAL_N}x${GLOBAL_N})..."
    
    BASELINE_TIME=0
    for r in $(seq 1 $REPEAT); do
        OUTPUT=$(mpirun -np 1 ./build/cg_prof $GLOBAL_N $MAX_ITER 2>&1)
        T=$(echo "$OUTPUT" | grep -oP 'Total:\s*\K[\d.]+' | head -1 || echo "0")
        BASELINE_TIME=$(echo "$BASELINE_TIME + $T" | bc)
    done
    BASELINE_TIME=$(echo "scale=6; $BASELINE_TIME / $REPEAT" | bc)
    
    for NP in $PROCESS_COUNTS; do
        # Calculate global problem size (maintaining constant local size)
        GRID_DIM=$(echo "scale=0; sqrt($NP)" | bc)
        GLOBAL_N=$(( LOCAL_N * GRID_DIM ))
        
        # Calculate nodes needed
        NODES=$(( (NP + 19) / 20 ))
        
        echo ""
        echo "np=$NP, global=${GLOBAL_N}x${GLOBAL_N}, nodes=$NODES"
        
        # Run and average
        TOTAL_TIME=0
        STENCIL_TIME=0
        HALO_TIME=0
        ALLREDUCE_TIME=0
        
        for r in $(seq 1 $REPEAT); do
            OUTPUT=$(mpirun -np $NP ./build/cg_prof $GLOBAL_N $MAX_ITER 2>&1)
            
            T=$(echo "$OUTPUT" | grep -oP 'Total:\s*\K[\d.]+' | head -1 || echo "0")
            TS=$(echo "$OUTPUT" | grep -oP 'Stencil:\s*\K[\d.]+' | head -1 || echo "0")
            TH=$(echo "$OUTPUT" | grep -oP 'Halo:\s*\K[\d.]+' | head -1 || echo "0")
            TA=$(echo "$OUTPUT" | grep -oP 'Allreduce:\s*\K[\d.]+' | head -1 || echo "0")
            
            TOTAL_TIME=$(echo "$TOTAL_TIME + $T" | bc)
            STENCIL_TIME=$(echo "$STENCIL_TIME + $TS" | bc)
            HALO_TIME=$(echo "$HALO_TIME + $TH" | bc)
            ALLREDUCE_TIME=$(echo "$ALLREDUCE_TIME + $TA" | bc)
        done
        
        AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $REPEAT" | bc)
        AVG_STENCIL=$(echo "scale=6; $STENCIL_TIME / $REPEAT" | bc)
        AVG_HALO=$(echo "scale=6; $HALO_TIME / $REPEAT" | bc)
        AVG_ALLREDUCE=$(echo "scale=6; $ALLREDUCE_TIME / $REPEAT" | bc)
        
        ITER_TIME=$(echo "scale=6; $AVG_TIME / $MAX_ITER * 1000" | bc)
        OVERHEAD=$(echo "scale=2; ($AVG_TIME - $BASELINE_TIME) / $BASELINE_TIME * 100" | bc)
        
        echo "Time: ${AVG_TIME}s, Overhead: ${OVERHEAD}%"
        
        echo -e "${LOCAL_N}\t${GLOBAL_N}\t${NP}\t${NODES}\t${AVG_TIME}\t${ITER_TIME}\t${AVG_STENCIL}\t${AVG_HALO}\t${AVG_ALLREDUCE}\t${OVERHEAD}" >> $RESULT_FILE
    done
done

echo ""
echo "=========================================="
echo "Weak Scaling Experiment Complete"
echo "Results saved to: $RESULT_FILE"
echo "=========================================="
