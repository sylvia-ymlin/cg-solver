#!/bin/bash
#===============================================================================
# Run All Distributed Experiments
#
# This script submits all three experiments as separate SLURM jobs.
# Results will be collected in the results/ directory.
#===============================================================================

cd $(dirname $0)/..

echo "=========================================="
echo "Submitting Distributed Experiments"
echo "=========================================="

# Check if binary exists
if [ ! -f "build/cg_prof" ]; then
    echo "Error: build/cg_prof not found. Please compile first:"
    echo "  make clean && make"
    exit 1
fi

# Create results directory
mkdir -p results

# Submit experiments
echo ""
echo "1. Submitting Strong Scaling experiment..."
JOB1=$(sbatch distributed/run_strong_scaling.sbatch | awk '{print $4}')
echo "   Job ID: $JOB1"

echo ""
echo "2. Submitting Weak Scaling experiment..."
JOB2=$(sbatch distributed/run_weak_scaling.sbatch | awk '{print $4}')
echo "   Job ID: $JOB2"

echo ""
echo "3. Submitting Node Boundary Analysis..."
JOB3=$(sbatch distributed/run_node_boundary.sbatch | awk '{print $4}')
echo "   Job ID: $JOB3"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View results when complete:"
echo "  ls -la results/"
echo ""
echo "Analyze results:"
echo "  python3 distributed/analyze_results.py"
echo "=========================================="
