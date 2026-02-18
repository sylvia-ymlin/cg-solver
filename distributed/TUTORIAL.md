# Distributed Experiments Tutorial

## Overview

This tutorial guides you through running distributed MPI experiments on UPPMAX (or any SLURM cluster) to validate the theoretical predictions about communication overhead and optimization strategies.

## Theoretical Background

### Key Predictions to Validate

| Prediction | Expected Observation |
|------------|---------------------|
| **Node boundary effect** | Significant jump in `T_halo` and `T_allreduce` when crossing node boundaries |
| **Strong scaling limit** | Speedup plateaus due to communication overhead |
| **Weak scaling overhead** | Time increases with process count due to communication |
| **Pipelined CG potential** | May become effective in distributed setting (network latency dominant) |

### Performance Model

```
Single iteration cost:
T_iter = T_stencil + T_axpy + T_halo + T_allreduce

Where:
- T_stencil: Compute-bound (5-point stencil operation)
- T_axpy: Memory-bound (vector updates)
- T_halo: Communication-bound (halo exchange)
- T_allreduce: Latency-bound (global reduction)

In UMA:
  T_halo << T_stencil → Memory bandwidth is bottleneck

In Distributed:
  T_halo = O(n/√p) / BW_network + latency
  T_allreduce = O(log p) × latency
  → Network becomes bottleneck when T_compute < T_comm
```

---

## Prerequisites

### 1. UPPMAX Account
- Username and password
- 2FA (Google/Microsoft Authenticator)
- Project allocation (e.g., `uppmax2024-2-9`)

### 2. SSH Access
```bash
# Connect to rackham (login node)
ssh ymlin@rackham.uppmax.uu.se

# Or to pelle (if applicable)
ssh ymlin@pelle.uppmax.uu.se
```

### 3. Module System
```bash
# Check available compilers
module avail gcc

# Check available MPI implementations
module avail mpi

# Load required modules
module load gcc/12.2.0
module load openmpi/4.1.4
```

---

## Step-by-Step Guide

### Step 1: Prepare Your Environment

```bash
# SSH to UPPMAX
ssh ymlin@rackham.uppmax.uu.se

# Clone or update the repository
cd ~
git clone https://github.com/sylvia-ymlin/cg-solver.git
# OR if already exists:
cd cg-solver && git pull

# Navigate to project directory
cd cg-solver
```

### Step 2: Compile the Code

```bash
# Load modules
module load gcc/12.2.0
module load openmpi/4.1.4

# Compile
make clean
make

# Verify binary exists
ls -la build/cg_prof
```

### Step 3: Quick Test (Interactive)

Before submitting batch jobs, test interactively:

```bash
# Request interactive session
salloc -A uppmax2024-2-9 -n 4 -t 00:10:00

# Run quick test
mpirun -np 4 ./build/cg_prof 256 50

# Exit interactive session
exit
```

### Step 4: Submit Experiments

```bash
# Option A: Submit all experiments at once
cd distributed
bash run_all.sbatch

# Option B: Submit individual experiments
sbatch run_strong_scaling.sbatch
sbatch run_weak_scaling.sbatch
sbatch run_node_boundary.sbatch
```

### Step 5: Monitor Jobs

```bash
# Check queue status
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Watch output files
tail -f results/strong_scaling_<JOB_ID>.out
```

### Step 6: Analyze Results

```bash
# After jobs complete, check results
ls -la results/

# View summary
cat results/*.tsv

# Generate plots (requires matplotlib)
python3 distributed/analyze_results.py

# Plots will be in docs/distributed/
```

---

## Experiment Details

### Experiment 1: Strong Scaling

**Purpose**: Measure speedup and efficiency with fixed problem size.

**Configuration**:
- Fixed global grid: 4096 × 4096
- Process counts: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256
- 3 runs per configuration

**Key Metrics**:
```
Speedup S(p) = T(1) / T(p)
Efficiency E(p) = S(p) / p
```

**Expected Results**:
- Near-linear speedup for small p
- Plateau at higher p due to communication
- Efficiency drops below 80% at some threshold

### Experiment 2: Weak Scaling

**Purpose**: Measure overhead when keeping work per process constant.

**Configuration**:
- Local grid sizes: 128×128, 256×256, 512×512
- Process counts: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
- Global grid = local_grid × √p

**Key Metrics**:
```
Overhead % = (T(p) - T(1)) / T(1) × 100
```

**Expected Results**:
- Larger local size → lower overhead (computation masks communication)
- Small local sizes show significant overhead growth
- Identifies minimum computation threshold

### Experiment 3: Node Boundary Analysis

**Purpose**: Compare intra-node vs inter-node communication.

**Configuration**:
- Focus on process counts around node boundaries (20, 40, 60, 80 cores)
- Detailed timing breakdown

**Key Observations**:
- **np ≤ 20**: Single node, shared memory communication
- **np > 20**: Cross-node, network communication

**Expected Results**:
```
At node boundary (e.g., np=20 → np=25):
  - T_halo should jump significantly
  - T_allreduce should increase
  - Overall efficiency drop
```

---

## Modifying Experiments

### Change Problem Size

Edit the `.sbatch` files:

```bash
# In run_strong_scaling.sbatch
PROBLEM_SIZE=8192  # Change from 4096

# In run_weak_scaling.sbatch
LOCAL_SIZES="64 128 256 512"  # Add more configurations
```

### Change Resource Allocation

```bash
# Request more nodes
#SBATCH --nodes=1-32

# Change time limit
#SBATCH -t 02:00:00

# Change partition (if applicable)
#SBATCH -p devel
```

### Add New Process Counts

Edit the `PROCESS_COUNTS` variable in the scripts. **Important**: Must be perfect squares for 2D domain decomposition.

```bash
PROCESS_COUNTS="1 4 9 16 25 36 49 64 81 100 121 144 169 196 225 256 324 400"
```

---

## Troubleshooting

### Common Issues

**1. "Munge authentication error"**
```bash
# You're on login node, need compute node
salloc -A uppmax2024-2-9 -n 1 -t 00:10:00
```

**2. "Module not found"**
```bash
# Check available versions
module avail gcc
module avail openmpi

# Load correct version
module load gcc/11.3.0  # Adjust version
```

**3. "Perfect square required"**
```
Process count must be 1, 4, 9, 16, 25, 36, 49, 64, ...
(For 2D domain decomposition: √p must be integer)
```

**4. Job runs but no timing output**
```bash
# Check stderr
cat results/strong_scaling_<JOB_ID>.err

# May need to adjust output parsing
grep -oP 'Total:\s*\K[\d.]+' output.txt
```

### Debug Mode

```bash
# Run with verbose MPI output
mpirun --verbose -np 4 ./build/cg_prof 256 50

# Check MPI configuration
ompi_info
```

---

## Expected Timeline

| Step | Time |
|------|------|
| Setup & Compilation | ~5 min |
| Quick Interactive Test | ~5 min |
| Strong Scaling Job | ~30-60 min |
| Weak Scaling Job | ~30-60 min |
| Node Boundary Job | ~20-30 min |
| Analysis & Plotting | ~5 min |

**Total**: ~2-3 hours (including queue wait)

---

## Interpreting Results

### Strong Scaling Analysis

```
If efficiency drops quickly:
  → Communication overhead dominates
  → Consider larger problem sizes or fewer processes

If efficiency stays high:
  → Good scalability
  → Memory bandwidth likely still the bottleneck
```

### Weak Scaling Analysis

```
If overhead < 10%:
  → Local computation effectively masks communication
  → n_local is above threshold

If overhead > 50%:
  → Communication dominates
  → Need larger local problem size
```

### Node Boundary Analysis

```
Compare halo_time ratio at:
  np=20 (single node) vs np=25 (cross-node)

If significant jump:
  → Validates network latency impact
  → Pipelined CG may be worth revisiting

If minimal jump:
  → Network bandwidth similar to memory bandwidth
  → Results similar to UMA environment
```

---

## Next Steps

After completing the distributed experiments:

1. **Compare with UMA results**:
   - How do the bottleneck characteristics differ?
   - Was the prediction about Pipelined CG correct?

2. **Document findings**:
   - Add results to your project report
   - Include plots and analysis

3. **Consider extensions**:
   - Test different networks (InfiniBand vs Ethernet)
   - Compare MPI implementations (OpenMPI vs MPICH)
   - Implement and test Pipelined CG variant

---

## Files Reference

```
distributed/
├── run_all.sbatch              # Submit all experiments
├── run_strong_scaling.sbatch   # Strong scaling experiment
├── run_weak_scaling.sbatch     # Weak scaling experiment
├── run_node_boundary.sbatch    # Node boundary analysis
├── analyze_results.py          # Generate plots from results
└── TUTORIAL.md                 # This file
```

---

## Contact & Support

- UPPMAX Support: support@uppmax.uu.se
- UPPMAX Documentation: https://docs.uppmax.uu.se/
- SNIC Documentation: https://docs.snic.se/

Good luck with your experiments! 🚀
