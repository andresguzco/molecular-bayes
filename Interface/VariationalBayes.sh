#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm Interface/main_QM9.py \
	--output-dir 'results/models/VI' \
	--model-name 'variational' \
	--data-path 'datasets/qm9' \
	--model-type 'Variational' \
	--epochs 100
