#!/bin/bash

# Loading the required module
sbatch slurm_launcher.slrm main_QM9.py \
    --output-dir 'results/models/laplace' \
    --model-name 'equiformer_v2' \
    --data-path 'datasets/qm9' \
    --model-type 'Laplace' \
    --epochs 100 \
    --epochs-bayes 100
