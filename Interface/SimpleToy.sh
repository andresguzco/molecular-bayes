#!/bin/bash

# Loading the required module
sbatch Interface/slurm_launcher.slrm Interace/main_QM9.py \
	--output-dir 'results/models/base' \
	--model-name 'equiformer_v2' \
	--data-path 'datasets/qm9' \
	--epochs 100

