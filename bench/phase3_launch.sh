#!/usr/bin/env bash
# Phase 3: launch the 9 training runs (3 arms x 3 seeds) side by side on the 2-socket EPYC 7742 host,
# each pinned to 14 physical cores plus their SMT siblings (28 logical CPUs), one socket per run
# where possible (run 8 takes the leftover 8 + 8 cores of both sockets).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
mkdir -p results/phase3/logs
cpus() { local a=$1 b=$2; echo "$a-$b,$((a+128))-$((b+128))"; }
SETS=("$(cpus 0 13)" "$(cpus 14 27)" "$(cpus 28 41)" "$(cpus 42 55)" "$(cpus 64 77)" "$(cpus 78 91)" "$(cpus 92 105)" "$(cpus 106 119)" "56-63,120-127,184-191,248-255")
k=0
for seed in 0 1 2; do
  for arm in baseline augmented equivariant; do
    OMP_NUM_THREADS=28 taskset -c "${SETS[$k]}" "$PY" bench/phase3_train.py "$arm" "$seed" --threads 28 \
      > "results/phase3/logs/train_${arm}_s${seed}.log" 2>&1 &
    k=$((k+1))
  done
done
wait
