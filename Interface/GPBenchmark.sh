#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm GP.py \
	--output-dir 'results/models/GP' \
	--model-name 'GP_Exact' \
	--model-type 'GP' \
	--data-path 'datasets/qm9' \
	--iterations 100 \