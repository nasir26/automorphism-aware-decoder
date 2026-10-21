#!/usr/bin/env bash
# Phase 4: QAT fine-tuning of baseline and equivariant models (3 seeds x 3 precisions = 18 runs) in
# two waves of 9 on the Phase 3 CPU sets, then the 18 golden-model evaluations the same way.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
mkdir -p results/phase4/logs
cpus() { local a=$1 b=$2; echo "$a-$b,$((a+128))-$((b+128))"; }
SETS=("$(cpus 0 13)" "$(cpus 14 27)" "$(cpus 28 41)" "$(cpus 42 55)" "$(cpus 64 77)" "$(cpus 78 91)" "$(cpus 92 105)" "$(cpus 106 119)" "56-63,120-127,184-191,248-255")
JOBS=()
for arm in equivariant baseline; do for seed in 0 1 2; do for prec in int8 int4 ternary; do JOBS+=("$arm $seed $prec"); done; done; done
run_wave() {  # script, logprefix, first index
  local script=$1 pre=$2 first=$3 k=0
  for j in "${JOBS[@]:$first:9}"; do
    set -- $j
    OMP_NUM_THREADS=28 taskset -c "${SETS[$k]}" "$PY" "bench/$script" "$1" "$2" "$3" --threads 28 \
      > "results/phase4/logs/${pre}_$1_s$2_$3.log" 2>&1 &
    k=$((k+1))
  done
  wait
}
run_wave phase4_qat.py qat 0
run_wave phase4_qat.py qat 9
run_wave phase4_eval.py eval 0
run_wave phase4_eval.py eval 9
