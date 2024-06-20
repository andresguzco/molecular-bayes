#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm main.py \
	--output-dir 'results/models/base' \
	--model-name 'equiformer_v2' \
	--data-path 'datasets/qm9' \
	--epochs 1

