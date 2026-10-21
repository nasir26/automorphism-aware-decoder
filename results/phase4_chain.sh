#!/usr/bin/env bash
# waits for the Phase 3 chain to finish its report, then runs Phase 4
cd "$(dirname "$0")/.."
until grep -q -E "evaluation and report done|ABORT" results/phase3/logs/chain.log 2>/dev/null; do sleep 60; done
if grep -q ABORT results/phase3/logs/chain.log; then echo "$(date) Phase 3 chain aborted; Phase 4 not started"; exit 1; fi
echo "$(date) Phase 4 started"
bench/phase4_launch.sh
echo "$(date) Phase 4 done"
