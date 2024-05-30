#!/bin/bash

# Loading the required module
sbatch slurm_launcher.slrm main_QM9.py \
    --output-dir 'results/models/toy' \
    --model-name 'equiformer_v2' \
    --data-path 'datasets/qm9' \
    --epochs 100