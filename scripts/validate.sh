#!/bin/bash
# Quick correctness test for the CG solver.
# Checks serial and parallel runs converge with matching results.
#
# L2 error threshold: 1e-8
#   The exact solution u = x(1-x)y(1-y) is quadratic, so the 5-point stencil
#   has zero truncation error. The only error comes from the iterative solver.
#   With tol=1e-10 and cond(A) ~ n^2 = 4096 (n=64), the expected L2 error
#   is h * ||A^{-1}|| * tol ~ (1/65) * 214 * 1e-10 ~ 3e-10, well below 1e-8.
set -e

echo "=== Building ==="
make clean && make

check_error() {
  local out="$1" label="$2" threshold="$3"
  echo "$out" | grep -q "Iterations:" || { echo "FAIL [$label]: did not complete"; exit 1; }
  echo "$out" | awk -v label="$label" -v thr="$threshold" '
    /L2 Error:/ {
      if ($3 + 0 > thr) { print "FAIL [" label "]: L2 error " $3 " exceeds threshold " thr; exit 1 }
      else               { print "PASS [" label "]: L2 error = " $3 }
    }'
}

echo ""
echo "=== Serial correctness (n=64, tol=1e-10) ==="
OUT=$(mpirun --oversubscribe -np 1 ./solver 64 10000 1e-10 0)
echo "$OUT"
check_error "$OUT" "serial" "1e-8"

echo ""
echo "=== Parallel correctness (np=4, n=64, tol=1e-10) ==="
OUT=$(mpirun --oversubscribe -np 4 ./solver 64 10000 1e-10 0)
echo "$OUT"
check_error "$OUT" "parallel" "1e-8"

echo ""
echo "=== Input validation ==="
mpirun --oversubscribe -np 1 ./solver -1 2>&1 | grep -q "Usage" || { echo "FAIL: did not reject n=-1"; exit 1; }
echo "PASS: rejects n <= 0"

echo ""
echo "=== Grid refinement study (problem=1: sin exact solution) ==="
# Verify O(h^2) convergence: error should decrease by ~4x each time n doubles.
prev_err=""
for n in 16 32 64 128; do
  OUT=$(mpirun --oversubscribe -np 1 ./solver $n 50000 1e-12 0 1)
  err=$(echo "$OUT" | awk '/L2 Error:/ { print $3+0 }')
  echo "n=$n: L2 error = $err"
  if [ -n "$prev_err" ]; then
    result=$(awk -v p="$prev_err" -v c="$err" -v n="$n" 'BEGIN {
      ratio = p/c
      printf "  ratio = %.2f", ratio
      if (ratio < 3.5 || ratio > 4.5) { print " FAIL: expected ~4 for O(h^2)"; exit 1 }
      else                             { print " PASS" }
    }')
    echo "$result" || { exit 1; }
  fi
  prev_err=$err
done

echo ""
echo "All tests passed."
