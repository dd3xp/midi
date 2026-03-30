#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "exp055: Encoder Fine-tuning for Amp Prediction"
echo "Started at: $(date)"
echo "=========================================="

# Create checkpoint and results directories
mkdir -p experiments/checkpoints/exp055
mkdir -p experiments/results

# Copy required checkpoints
echo "Copying baseline checkpoint..."
cp -n experiments/checkpoints/exp033/baseline_best.pt experiments/checkpoints/exp055/baseline_best.pt 2>/dev/null || true

echo "Copying diffusion checkpoint..."
cp -n experiments/checkpoints/exp034/diffusion_best_ema.pt experiments/checkpoints/exp055/diffusion_best_ema.pt 2>/dev/null || true

# Train (stage 2: diffusion frozen, encoder unfrozen, amp predictor training)
echo ""
echo "=========================================="
echo "Training: encoder unfrozen (lr=2e-6) + amp predictor (lr=5e-5)"
echo "=========================================="
python src/model/train.py \
    --config experiments/configs/exp055.yaml \
    --stage 2

# Evaluate
echo ""
echo "=========================================="
echo "Evaluating..."
echo "=========================================="
python src/model/evaluate.py \
    --checkpoint_dir experiments/checkpoints/exp055/ \
    --config experiments/configs/exp055.yaml \
    --mode both \
    --n_samples 5 \
    --ddim_steps 50 \
    --eta 0.3 \
    --max_eval_len 8192 \
    --output experiments/results/exp055.json

echo ""
echo "=========================================="
echo "exp055: All done at: $(date)"
echo "Results: experiments/results/exp055.json"
echo "=========================================="
