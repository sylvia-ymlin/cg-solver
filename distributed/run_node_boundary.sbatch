#!/bin/bash
#===============================================================================
# Node Boundary Analysis for MPI CG Solver
#
# Purpose: Compare performance characteristics within vs across nodes
#          to understand the impact of network communication.
#
# This is the KEY experiment for validating theoretical predictions:
#   - Within node: shared memory (low latency, high bandwidth)
#   - Across nodes: network (higher latency, lower bandwidth)
#
# Expected observation:
#   - Significant jump in T_halo and T_allreduce at node boundaries
#   - Potential for Pipelined CG to become effective in distributed setting
#===============================================================================

#SBATCH -A uppmax2024-2-9
#SBATCH -J cg_boundary
#SBATCH -t 00:45:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH -o results/node_boundary_%j.out
#SBATCH -e results/node_boundary_%j.err

# Load modules
module load gcc/12.2.0
module load openmpi/4.1.4

cd $SLURM_SUBMIT_DIR
mkdir -p results

# Configuration
PROBLEM_SIZE=2048
MAX_ITER=100
REPEAT=5

# Output file
RESULT_FILE="results/node_boundary_$(date +%Y%m%d_%H%M%S).tsv"
echo "Node Boundary Analysis - $(date)" > $RESULT_FILE
echo -e "np\tnodes\tconfig\ttime_s\tstencil_ms\thalo_ms\tallreduce_ms\taxpy_ms\tprecond_ms\thalo_ratio" >> $RESULT_FILE

echo "=========================================="
echo "Node Boundary Analysis"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Cores per node: 20"
echo "=========================================="

# Configurations to test:
# Format: np:description
# Focus on boundary points: 16, 20, 21 (crossing single node)
#                           36, 40, 41 (crossing two nodes)
#                           60, 80, 81 (crossing four nodes)

CONFIGS=(
    "1:single_proc"
    "4:single_node_small"
    "9:single_node_medium"
    "16:single_node_large"
    "20:single_node_full"
    "25:cross_node_1node+5"
    "36:almost_2nodes"
    "40:two_nodes_full"
    "64:three_nodes+"
    "80:four_nodes_full"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r NP DESC <<< "$CONFIG"
    
    # Calculate nodes needed
    NODES=$(( (NP + 19) / 20 ))
    
    echo ""
    echo "=========================================="
    echo "np=$NP ($DESC), nodes=$NODES"
    echo "=========================================="
    
    TOTAL_TIME=0
    STENCIL_TIME=0
    HALO_TIME=0
    ALLREDUCE_TIME=0
    AXPY_TIME=0
    PRECOND_TIME=0
    
    for r in $(seq 1 $REPEAT); do
        OUTPUT=$(mpirun -np $NP ./build/cg_prof $PROBLEM_SIZE $MAX_ITER 2>&1)
        
        T=$(echo "$OUTPUT" | grep -oP 'Total:\s*\K[\d.]+' | head -1 || echo "0")
        TS=$(echo "$OUTPUT" | grep -oP 'Stencil:\s*\K[\d.]+' | head -1 || echo "0")
        TH=$(echo "$OUTPUT" | grep -oP 'Halo:\s*\K[\d.]+' | head -1 || echo "0")
        TA=$(echo "$OUTPUT" | grep -oP 'Allreduce:\s*\K[\d.]+' | head -1 || echo "0")
        TX=$(echo "$OUTPUT" | grep -oP 'AXPY:\s*\K[\d.]+' | head -1 || echo "0")
        TP=$(echo "$OUTPUT" | grep -oP 'Precond:\s*\K[\d.]+' | head -1 || echo "0")
        
        TOTAL_TIME=$(echo "$TOTAL_TIME + $T" | bc)
        STENCIL_TIME=$(echo "$STENCIL_TIME + $TS" | bc)
        HALO_TIME=$(echo "$HALO_TIME + $TH" | bc)
        ALLREDUCE_TIME=$(echo "$ALLREDUCE_TIME + $TA" | bc)
        AXPY_TIME=$(echo "$AXPY_TIME + $TX" | bc)
        PRECOND_TIME=$(echo "$PRECOND_TIME + $TP" | bc)
    done
    
    AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $REPEAT" | bc)
    AVG_STENCIL=$(echo "scale=6; $STENCIL_TIME / $REPEAT" | bc)
    AVG_HALO=$(echo "scale=6; $HALO_TIME / $REPEAT" | bc)
    AVG_ALLREDUCE=$(echo "scale=6; $ALLREDUCE_TIME / $REPEAT" | bc)
    AVG_AXPY=$(echo "scale=6; $AXPY_TIME / $REPEAT" | bc)
    AVG_PRECOND=$(echo "scale=6; $PRECOND_TIME / $REPEAT" | bc)
    
    # Calculate halo time ratio (communication overhead)
    HALO_RATIO=$(echo "scale=4; $AVG_HALO / $AVG_TIME * 100" | bc)
    
    echo "Time: ${AVG_TIME}s, Stencil: ${AVG_STENCIL}s, Halo: ${AVG_HALO}s (${HALO_RATIO}%)"
    
    echo -e "${NP}\t${NODES}\t${DESC}\t${AVG_TIME}\t${AVG_STENCIL}\t${AVG_HALO}\t${AVG_ALLREDUCE}\t${AVG_AXPY}\t${AVG_PRECOND}\t${HALO_RATIO}" >> $RESULT_FILE
done

echo ""
echo "=========================================="
echo "Node Boundary Analysis Complete"
echo "Results saved to: $RESULT_FILE"
echo ""
echo "Key Analysis Points:"
echo "  - Compare halo_ratio at np=20 (single node) vs np=25 (cross-node)"
echo "  - Observe communication overhead jump at node boundaries"
echo "=========================================="
