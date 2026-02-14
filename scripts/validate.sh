#!/bin/bash
# Quick correctness test for the CG solver.
# Checks serial and parallel runs converge with matching results.
set -e

echo "=== Building ==="
make clean && make

echo ""
echo "=== Serial correctness (n=64, tol=1e-10) ==="
OUT=$(mpirun --oversubscribe -np 1 ./CG 64 10000 1e-10)
echo "$OUT"
echo "$OUT" | grep -q "Converged: yes" || { echo "FAIL: did not converge"; exit 1; }

echo ""
echo "=== Parallel correctness (np=4, n=64, tol=1e-10) ==="
OUT=$(mpirun --oversubscribe -np 4 ./CG 64 10000 1e-10)
echo "$OUT"
echo "$OUT" | grep -q "Converged: yes" || { echo "FAIL: did not converge"; exit 1; }

echo ""
echo "=== Input validation ==="
mpirun --oversubscribe -np 1 ./CG -1 2>&1 | grep -q "Error" && echo "PASS: rejects n=-1"
mpirun --oversubscribe -np 4 ./CG 1 2>&1 | grep -q "Error" && echo "PASS: rejects n < sqrt(p)"

echo ""
echo "All tests passed."
