#!/bin/bash -l
# NOTE: You must use an allocation/project account that you are a member of.
# Check your projects with e.g. `projinfo $USER` / `projmembers`, then submit with:
#   sbatch -A <project> scripts/uppmax_bench.sbatch
#SBATCH -A uppmax2025-2-316
#SBATCH -p pelle
#SBATCH -J cg-bench
#SBATCH -t 00:20:00
#SBATCH -n 64
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -x  # Enable debug logging
# set -euo pipefail  # Disable strict mode for debugging

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_NUM_THREADS=1

timestamp="$(date +%Y%m%d_%H%M%S)"
outdir="results/uppmax_${timestamp}"
mkdir -p "${outdir}"

{
  echo "timestamp=${timestamp}"
  echo "pwd=$(pwd)"
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "nodes=${SLURM_JOB_NUM_NODES:-}"
  echo "ntasks=${SLURM_NTASKS:-}"
  echo "submit_host=${SLURM_SUBMIT_HOST:-}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
  echo "date=$(date -Is)"
} | tee "${outdir}/job_env.txt"

MODULES=""                 # Rely on manual 'module load'
MODULE_PURGE="0"           # Do not purge loaded modules

if type module >/dev/null 2>&1; then
  if [[ "${MODULE_PURGE}" == "1" ]]; then
    module purge || true
  fi

  if [[ -n "${MODULES}" ]]; then
    echo "Loading modules from MODULES='${MODULES}'" | tee -a "${outdir}/modules_load.txt"
    # shellcheck disable=SC2086
    module load ${MODULES} 2>&1 | tee -a "${outdir}/modules_load.txt"
  else
    # Best-effort defaults for UPPMAX (names vary by system)
    # For best reproducibility, replace with explicit versions after:
    #   module spider GCC; module spider OpenMPI; module spider Python
    if module load GCC OpenMPI >/dev/null 2>&1; then
      echo "Loaded modules: GCC OpenMPI" | tee -a "${outdir}/modules_load.txt"
    elif module load foss >/dev/null 2>&1; then
      echo "Loaded modules: foss" | tee -a "${outdir}/modules_load.txt"
    elif module load gcc openmpi >/dev/null 2>&1; then
      echo "Loaded modules: gcc openmpi" | tee -a "${outdir}/modules_load.txt"
    else
      echo "WARNING: could not load an MPI toolchain module; relying on default environment." | tee -a "${outdir}/modules_load.txt"
      echo "Hint: run 'module spider GCC' / 'module spider OpenMPI' and submit with MODULES='...'" | tee -a "${outdir}/modules_load.txt"
    fi

    if module load Python >/dev/null 2>&1; then
      echo "Loaded module: Python" | tee -a "${outdir}/modules_load.txt"
    elif module load python3 >/dev/null 2>&1; then
      echo "Loaded module: python3" | tee -a "${outdir}/modules_load.txt"
    else
      echo "WARNING: could not load a Python module; relying on default python3." | tee -a "${outdir}/modules_load.txt"
    fi
  fi

  module list 2>&1 | tee "${outdir}/modules.txt" || true
else
  echo "No modules system detected (no 'module' function). Skipping module load." | tee -a "${outdir}/modules_load.txt"
fi

{
  command -v mpicc >/dev/null 2>&1 && mpicc --version || true
  command -v mpirun >/dev/null 2>&1 && mpirun --version || true
  command -v srun >/dev/null 2>&1 && srun --version || true
} | tee "${outdir}/tool_versions.txt"

if ! command -v mpicc >/dev/null 2>&1; then
  echo "ERROR: mpicc not found. Load an MPI module (e.g. OpenMPI) or use the correct UPPMAX toolchain for your system." | tee -a "${outdir}/progress.txt"
  exit 2
fi

echo "=== build ===" | tee "${outdir}/build.txt"
make clean >>"${outdir}/build.txt" 2>&1 || true
make >>"${outdir}/build.txt" 2>&1

NPS=(1 4 9 16 25 36 49 64)
declare -A SQRTP=([1]=1 [4]=2 [9]=3 [16]=4 [25]=5 [36]=6 [49]=7 [64]=8)

# Strong scaling config (fixed global n)
STRONG_N="${STRONG_N:-2048}"
STRONG_ITERS="${STRONG_ITERS:-200}"

# Weak scaling config (fixed local_n per rank; global n = local_n * sqrt(p))
WEAK_LOCAL_N="${WEAK_LOCAL_N:-512}"
WEAK_ITERS="${WEAK_ITERS:-200}"

SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"     # e.g. "--mpi=pmix"
MPIRUN_EXTRA_ARGS="${MPIRUN_EXTRA_ARGS:-}" # e.g. "--bind-to core"

run_mpi() {
  local np="$1"; shift
  # Always use mpirun on UPPMAX with foss/OpenMPI to avoid srun conflicts
  mpirun -np "${np}" ${MPIRUN_EXTRA_ARGS} "$@"
}

extract_field() {
  # Simple robust extraction: find line with "Key:", print next token
  local key="$1"
  awk -v k="${key}" '$1 == k {print $2; exit}'
}

run_and_log() {
  local tag="$1"
  local grid="$2"
  local iters="$3"
  local np="$4"
  local outfile="$5"

  echo "=== ${tag}: grid=${grid} iters=${iters} np=${np} ===" | tee -a "${outdir}/progress.txt"

  set +e
  out="$(run_mpi "${np}" ./CG "${grid}" "${iters}" 0 2>&1)"
  rc=$?
  set -e

  printf "%s\n" "${out}" > "${outfile}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "RUN_FAILED tag=${tag} grid=${grid} iters=${iters} np=${np} rc=${rc}" | tee -a "${outdir}/progress.txt"
    return 0
  fi

  # CG prints these on rank 0; parse from the saved output file to avoid any
  # command-substitution quirks and handle launcher-added prefixes.
  total_time="$(extract_field "Time:" < "${outfile}" || true)"
  time_halo="$(extract_field "Time_Halo:" < "${outfile}" || true)"
  time_reduce="$(extract_field "Time_Reduce:" < "${outfile}" || true)"
  time_comp="$(extract_field "Time_Comp:" < "${outfile}" || true)"
  iters_done="$(extract_field "Iterations:" < "${outfile}" || true)"

  if [[ -z "${total_time}" || -z "${iters_done}" ]]; then
    echo "PARSE_FAILED tag=${tag} grid=${grid} iters=${iters} np=${np}" | tee -a "${outdir}/progress.txt"
    return 0
  fi

  iter_time_ms="$(awk -v t="${total_time}" -v it="${iters}" 'BEGIN{printf "%.6f", 1000.0*t/it}')"
  halo_ms="$(awk -v t="${time_halo:-0}" -v it="${iters}" 'BEGIN{printf "%.6f", 1000.0*t/it}')"
  reduce_ms="$(awk -v t="${time_reduce:-0}" -v it="${iters}" 'BEGIN{printf "%.6f", 1000.0*t/it}')"
  comp_ms="$(awk -v t="${time_comp:-0}" -v it="${iters}" 'BEGIN{printf "%.6f", 1000.0*t/it}')"

  # Parse-friendly single-line record (iter_time is in ms/iter)
  echo "grid=${grid} np=${np} iters=${iters} total_time=${total_time} iter_time=${iter_time_ms} halo_time=${time_halo:-} reduce_time=${time_reduce:-} comp_time=${time_comp:-} halo_ms=${halo_ms} reduce_ms=${reduce_ms} comp_ms=${comp_ms}" \
    >> "${outdir}/${tag}.log"

  # Append TSV rows directly (more robust than tailing the .log file).
  if [[ "${tag}" == "strong_scaling" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${grid}" "${np}" "${iters}" "${total_time}" "${iter_time_ms}" "${halo_ms}" "${reduce_ms}" "${comp_ms}" \
      >> "${outdir}/strong_scaling.tsv"
  elif [[ "${tag}" == "weak_scaling" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${WEAK_LOCAL_N}" "${grid}" "${np}" "${iters}" "${total_time}" "${iter_time_ms}" "${halo_ms}" "${reduce_ms}" "${comp_ms}" \
      >> "${outdir}/weak_scaling.tsv"
  fi
}

echo -e "grid\tnp\titers\ttotal_time_s\titer_time_ms\thalo_ms\treduce_ms\tcomp_ms" > "${outdir}/strong_scaling.tsv"
echo -e "local_n\tgrid\tnp\titers\ttotal_time_s\titer_time_ms\thalo_ms\treduce_ms\tcomp_ms" > "${outdir}/weak_scaling.tsv"

echo "=== strong scaling ===" | tee -a "${outdir}/progress.txt"
for np in "${NPS[@]}"; do
  if [[ -n "${SLURM_NTASKS:-}" ]] && [[ "${np}" -gt "${SLURM_NTASKS}" ]]; then
    echo "SKIP strong np=${np} (allocated ntasks=${SLURM_NTASKS})" | tee -a "${outdir}/progress.txt"
    continue
  fi

  run_and_log "strong_scaling" "${STRONG_N}" "${STRONG_ITERS}" "${np}" "${outdir}/strong_np${np}.out"
done

echo "=== weak scaling ===" | tee -a "${outdir}/progress.txt"
for np in "${NPS[@]}"; do
  if [[ -n "${SLURM_NTASKS:-}" ]] && [[ "${np}" -gt "${SLURM_NTASKS}" ]]; then
    echo "SKIP weak np=${np} (allocated ntasks=${SLURM_NTASKS})" | tee -a "${outdir}/progress.txt"
    continue
  fi

  sqrtp="${SQRTP[${np}]:-}"
  if [[ -z "${sqrtp}" ]]; then
    echo "SKIP weak np=${np} (not a perfect square in SQRTP map)" | tee -a "${outdir}/progress.txt"
    continue
  fi

  grid="$((WEAK_LOCAL_N * sqrtp))"
  run_and_log "weak_scaling" "${grid}" "${WEAK_ITERS}" "${np}" "${outdir}/weak_np${np}.out"
done

echo "DONE outdir=${outdir}" | tee -a "${outdir}/progress.txt"
