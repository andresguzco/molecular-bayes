#!/bin/bash

# Loading the required module
sbatch slurm_launcher.slrm main_QM9.py \
    --output-dir 'results/models/VI' \
    --model-name 'variational' \
    --data-path 'datasets/qm9' \
    --variational True \
    --epochs 100
