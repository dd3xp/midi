#!/bin/bash
# Automated experiment loop: worker runs experiment -> supervisor reviews -> repeat
# Usage: bash run_experiments.sh [max_rounds]
# Stop anytime with Ctrl+C, results are saved in experiments/log.md

MAX_ROUNDS=${1:-10}
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting experiment loop (max $MAX_ROUNDS rounds)"
echo "Results will be logged to experiments/log.md"
echo "Press Ctrl+C to stop"
echo ""

for i in $(seq 1 $MAX_ROUNDS); do
    echo "========================================"
    echo "Round $i / $MAX_ROUNDS"
    echo "========================================"

    # Worker: run next experiment
    echo "[Worker] Running experiment..."
    claude --agent worker --print \
        "Read experiments/log.md. If code is not yet implemented, implement all modules first. \
         Then run the next experiment following the Experiment Loop Protocol. \
         Log results to experiments/log.md." \
        --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
        2>&1 | tee "experiments/worker_round_${i}.log"

    # Supervisor: review and plan next
    echo "[Supervisor] Reviewing results..."
    claude --agent supervisor --print \
        "Review the latest experiment in experiments/log.md. \
         Check code correctness if new code was written. \
         Analyze results, diagnose issues, and write next experiment plan to experiments/log.md." \
        --allowedTools "Read,Grep,Glob,Bash" \
        2>&1 | tee "experiments/supervisor_round_${i}.log"

    echo ""
    echo "Round $i complete. Check experiments/log.md for results."
    echo ""
done

echo "Experiment loop finished after $MAX_ROUNDS rounds."
echo "Check experiments/log.md for all results."
