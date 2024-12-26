#!/bin/bash
set -e

# Train models
echo "Training standard model..."
python train.py --model standard --epochs 10 --batch-size 64 --lr 0.001

echo "Training ternary model..."
python train.py --model ternary --epochs 10 --batch-size 64 --lr 0.001

# Run comparison
echo "Comparing models..."
python compare.py --model-dir ./models --results-dir ./results

# Generate visualization
echo "Generating plots..."
python plot_results.py --results-dir ./results