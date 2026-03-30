#!/bin/bash
# Automated experiment loop: worker runs experiment -> supervisor reviews -> repeat
# Usage: bash run_experiments.sh [max_rounds]
# Stop anytime with Ctrl+C, results are saved in experiments/log.md

MAX_ROUNDS=${1:-10}
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create directories needed by the experiment loop
mkdir -p experiments/configs experiments/checkpoints experiments/results

# Auto-detect starting round from existing log files
START_ROUND=1
for f in experiments/worker_prep_round_*.log experiments/train_round_*.log experiments/supervisor_round_*.log; do
    [ -f "$f" ] || continue
    num=$(echo "$f" | sed 's/.*round_\([0-9]*\)\.log/\1/')
    if [ "$num" -ge "$START_ROUND" ]; then
        START_ROUND=$((num + 1))
    fi
done
END_ROUND=$((START_ROUND + MAX_ROUNDS - 1))

echo "================================================"
echo "  MIDI-to-Expression Experiment Loop"
echo "  Rounds: $START_ROUND -> $END_ROUND ($MAX_ROUNDS rounds)"
echo "  Results: experiments/log.md"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

for i in $(seq $START_ROUND $END_ROUND); do
    echo "========================================"
    echo "  Round $i / $MAX_ROUNDS"
    echo "========================================"

    # Phase 1: Worker prepares experiment (config, code fixes)
    echo "[Worker] Preparing experiment..."
    claude --agent worker -p \
        "Read experiments/log.md and follow the '下一步计划'. \
         Step 1: Fix any code bugs mentioned in the plan. \
         Step 2: Create the experiment config YAML in experiments/configs/. \
         Step 3: Do NOT run training yet. Instead, write the exact training command \
         to a file: experiments/next_train_cmd.sh (a bash script that runs training \
         and then evaluation). Make sure the script includes: \
         - mkdir -p for the checkpoint directory \
         - python src/model/train.py with --config and --stage flags \
         - python src/model/evaluate.py after training completes \
         - The script should exit with 0 on success, non-zero on failure \
         Step 4: Print a brief summary of what you prepared." \
        --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
        --permission-mode bypassPermissions \
        2>&1 | tee "experiments/worker_prep_round_${i}.log"

    # Phase 2: Execute training (directly in shell, no timeout issues)
    if [ -f experiments/next_train_cmd.sh ]; then
        echo "[Training] Executing training script..."
        chmod +x experiments/next_train_cmd.sh
        bash experiments/next_train_cmd.sh 2>&1 | tee "experiments/train_round_${i}.log"
        TRAIN_EXIT=$?
        echo "[Training] Finished with exit code: $TRAIN_EXIT"
    else
        echo "[ERROR] Worker did not create experiments/next_train_cmd.sh"
        TRAIN_EXIT=1
    fi

    # Phase 3: Worker logs results
    echo "[Worker] Logging results..."
    claude --agent worker -p \
        "Training for this round has finished (exit code: $TRAIN_EXIT). \
         Read the training log at experiments/train_round_${i}.log (last 50 lines). \
         Read the evaluation results JSON if it exists. \
         Append a row to experiments/log.md results table. \
         Update '当前最佳' if this is the best result so far. \
         Do NOT write '下一步计划' — the supervisor will do that. \
         If training failed, note the failure in the log." \
        --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
        --permission-mode bypassPermissions \
        2>&1 | tee "experiments/worker_log_round_${i}.log"

    # Phase 4: Supervisor reviews and plans next
    echo "[Supervisor] Reviewing results..."
    claude --agent supervisor -p \
        "Review the latest experiment in experiments/log.md. \
         Read the latest config in experiments/configs/ and results in experiments/results/. \
         If new code was written, check correctness (read the code, grep for issues). \
         Analyze results, diagnose issues, and write the next experiment plan \
         to experiments/log.md under '下一步计划'. \
         If you find code bugs, list them clearly in the plan for the worker to fix." \
        --allowedTools "Read,Grep,Glob,Bash" \
        --permission-mode bypassPermissions \
        2>&1 | tee "experiments/supervisor_round_${i}.log"

    # Phase 5: Check if supervisor requests human intervention
    if [ -f experiments/NEEDS_HUMAN.txt ]; then
        echo ""
        echo "================================================"
        echo "  ⚠ SUPERVISOR REQUESTS HUMAN INTERVENTION"
        echo "================================================"
        cat experiments/NEEDS_HUMAN.txt
        echo ""
        echo "Review the issue above, then delete experiments/NEEDS_HUMAN.txt to continue."
        echo "Waiting..."
        while [ -f experiments/NEEDS_HUMAN.txt ]; do
            sleep 10
        done
        echo "NEEDS_HUMAN.txt removed. Resuming..."
    fi

    # Phase 6: Backup source code for reproducibility
    LATEST_EXP=$(ls -1 experiments/configs/ | sort | tail -1 | sed 's/\.yaml//')
    if [ -n "$LATEST_EXP" ]; then
        BACKUP_DIR="experiments/checkpoints/${LATEST_EXP}/src_backup"
        echo "[Backup] Saving source code to ${BACKUP_DIR}..."
        rm -rf "$BACKUP_DIR"
        mkdir -p "$BACKUP_DIR/model" "$BACKUP_DIR/data"
        cp src/model/*.py "$BACKUP_DIR/model/" 2>/dev/null
        cp src/data/*.py "$BACKUP_DIR/data/" 2>/dev/null
    fi

    echo ""
    echo "Round $i complete. Check experiments/log.md for results."
    echo ""
done

echo "================================================"
echo "  Experiment loop finished after $MAX_ROUNDS rounds."
echo "  Check experiments/log.md for all results."
echo "================================================"
