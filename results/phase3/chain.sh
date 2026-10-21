#!/usr/bin/env bash
# waits for the 9 training runs, checks each finished 20 epochs, then starts the evaluation
cd "$(dirname "$0")/../.."
while pgrep -f "bench/phase3_train.py" > /dev/null; do sleep 60; done
n=$(../.venv/bin/python -c "
import json,glob
print(sum(len(json.load(open(f))['history'])==20 for f in glob.glob('results/phase3/models/*/train.json')))")
echo "$(date) training done, complete runs: $n"
if [ "$n" != "9" ]; then echo "ABORT: not all 9 runs completed 20 epochs"; exit 1; fi
echo "$(date) evaluation started"
bench/phase3_eval_launch.sh
echo "$(date) evaluation and report done"
