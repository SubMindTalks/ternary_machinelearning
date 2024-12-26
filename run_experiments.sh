#!/bin/bash

# Activate Python environment
source ~/env/bin/activate

# Train and evaluate the Complex Number Model
echo "Training and evaluating the Complex Number Model..."
python train.py --model complex --epochs 5

# Train and evaluate the Ternary Logic Model
echo "Training and evaluating the Ternary Logic Model..."
python train.py --model ternary --epochs 5

# Evaluate models on augmented datasets
echo "Evaluating Ternary Logic Model on augmented datasets..."
python test_augmentations.py --model_path ternary_model.h5

# Deactivate environment
deactivate
