#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm main.py \
	--output-dir 'results/models/VI' \
	--model-name 'variational' \
	--data-path 'datasets/qm9' \
	--model-type 'Variational' \
	--epochs 20\
	--iterations 20 \
