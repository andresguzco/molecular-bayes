#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm main.py \
	--output-dir 'results/models/laplace' \
	--model-name 'equiformer_v2' \
	--data-path 'datasets/qm9' \
	--model-type 'Laplace' \
	--epochs 200 \
	--iterations 30 \
	--seed 1234 \
	--lmax 4
